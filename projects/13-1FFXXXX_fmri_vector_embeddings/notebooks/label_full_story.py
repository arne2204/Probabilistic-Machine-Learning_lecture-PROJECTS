#!/usr/bin/env python3
"""
Context-Aware Token Labeling using Anthropic API
Processes tokens with semantic context and generates probabilistic labels
"""

import pandas as pd
import json
import time
import os
from typing import List, Dict, Any
from anthropic import Anthropic
import argparse
from datetime import datetime
from dotenv import load_dotenv

class TokenLabeler:
    def __init__(self, api_key: str, max_context_tokens: int = 500):
        """
        Initialize the token labeler

        Args:
            api_key: Anthropic API key
            max_context_tokens: Maximum number of previous tokens to include as context
        """
        load_dotenv("./.env")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)
        self.max_context_tokens = max_context_tokens
        self.processed_count = 0

    def clean_tokens(self, df: pd.DataFrame) -> List[str]:
        """Clean and extract tokens from the CSV DataFrame"""
        print("Cleaning tokens...")

        # Get the first column (assuming it contains the tokens)
        column_name = df.columns[0]
        raw_tokens = df[column_name].tolist()

        # Clean tokens - remove the [""] wrapper
        clean_tokens = []
        for token in raw_tokens:
            if pd.isna(token) or token == '':
                continue

            # Remove [" and "] wrapper and handle escaped quotes
            cleaned = str(token).strip()
            if cleaned.startswith('["') and cleaned.endswith('"]'):
                cleaned = cleaned[2:-2]  # Remove [" and "]
                cleaned = cleaned.replace('""', '"')  # Handle escaped quotes

            if cleaned.strip():
                clean_tokens.append(cleaned.strip())

        print(f"Cleaned {len(clean_tokens)} tokens")
        return clean_tokens

    def get_context(self, tokens: List[str], index: int) -> Dict[str, Any]:
        """Get cumulative context (all previous tokens) for a specific token"""
        target_token = tokens[index]

        # Get all previous tokens (from start to current index, exclusive)
        all_previous = tokens[:index] if index > 0 else []

        # Optionally limit context length to avoid API token limits
        if len(all_previous) > self.max_context_tokens:
            context_before = all_previous[-self.max_context_tokens:]
        else:
            context_before = all_previous

        return {
            'target_token': target_token,
            'context_before': context_before,
            'context_after': [],  # Always empty in cumulative approach
            'full_context': ' '.join(context_before + [target_token])
        }

    def create_labeling_prompt(self, context_info: Dict[str, Any]) -> str:
        """Create a prompt for the API to label a token"""
        return f"""You are a semantic token labeling expert. Analyze the given token within its cumulative context (all previous story content) and provide labels for these four categories:

1. **location** - Physical places, geographical locations, spatial references
2. **characters** - People, character names, pronouns referring to people, roles
3. **emotions** - Emotional states, feelings, mood descriptors
4. **time** - Temporal references, time periods, time-related words

**Target Token:** "{context_info['target_token']}"

**Story Context (all previous tokens):** {' '.join(context_info['context_before']) if context_info['context_before'] else '[story beginning]'}

For each category, provide:
- **value**: The specific semantic label (e.g., "office", "Dr. Carmen Reed", "tired", "afternoon") or "null" if not applicable
- **confidence**: Float between 0.0-1.0 indicating your confidence in the label

Make sure the character you assign is one of the following and no other:
- Dr. Carmen Reed
- Dr. John Torreson
- Antonio
- Juan Torres
- Alba
- Maria
- Linda
- Ramiro
- Boat Driver
- Alba's Mother


Consider the cumulative story context when labeling. Use the full narrative context to understand character references, location continuity, emotional progression, and temporal flow.
Null values for labels can be assigned when no meaningful label can be found. This should only be done in exceptional cases. The confidence score for a null value should always be 0.0.

Respond in this exact JSON format:
{{
  "location": {{"value": "label_or_null", "confidence": 0.0}},
  "characters": {{"value": "label_or_null", "confidence": 0.0}},
  "emotions": {{"value": "label_or_null", "confidence": 0.0}},
  "time": {{"value": "label_or_null", "confidence": 0.0}}
}}"""

    def label_token(self, context_info: Dict[str, Any]) -> Dict[str, Any]:
        """Send a token to the API for labeling"""
        prompt = self.create_labeling_prompt(context_info)

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-latest",  # Use Haiku for cost efficiency
                max_tokens=300,
                temperature=0.1,  # Low temperature for consistent formatting
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract and parse JSON response
            response_text = response.content[0].text.strip()

            # Try to extract JSON from response
            try:
                # Look for JSON block
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]
                else:
                    raise ValueError("No JSON found in response")

                labels = json.loads(json_text)

                # Validate structure
                required_categories = ['location', 'characters', 'emotions', 'time']
                for category in required_categories:
                    if category not in labels:
                        labels[category] = {"value": "null", "confidence": 0.9}
                    elif 'value' not in labels[category] or 'confidence' not in labels[category]:
                        labels[category] = {"value": "null", "confidence": 0.9}

                return labels

            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON parsing error: {e}")
                print(f"Response text: {response_text}")
                # Return default structure on parsing error
                return {
                    "location": {"value": "null", "confidence": 0.9},
                    "characters": {"value": "null", "confidence": 0.9},
                    "emotions": {"value": "null", "confidence": 0.9},
                    "time": {"value": "null", "confidence": 0.9}
                }

        except Exception as e:
            print(f"API error for token '{context_info['target_token']}': {e}")
            # Return default structure on API error
            return {
                "location": {"value": "null", "confidence": 0.9},
                "characters": {"value": "null", "confidence": 0.9},
                "emotions": {"value": "null", "confidence": 0.9},
                "time": {"value": "null", "confidence": 0.9}
            }

    def process_tokens(self, tokens: List[str], batch_size: int = 1, delay: float = 0.5) -> List[Dict[str, Any]]:
        """Process all tokens with API labeling"""
        results = []
        total_tokens = len(tokens)

        print(f"Processing {total_tokens} tokens...")
        print(f"Using cumulative context (max {self.max_context_tokens} previous tokens)")
        print(f"API delay: {delay} seconds between calls")

        for i, token in enumerate(tokens):
            # Get context for this token
            context_info = self.get_context(tokens, i)

            # Get labels from API
            labels = self.label_token(context_info)

            # Create result entry
            result = {
                "token": token,
                "index": i,
                "labels": labels
            }

            results.append(result)
            self.processed_count += 1

            # Progress reporting
            if (i + 1) % 10 == 0 or i + 1 == total_tokens:
                print(f"Processed {i + 1}/{total_tokens} tokens ({((i + 1)/total_tokens)*100:.1f}%)")

            # Rate limiting
            if i < total_tokens - 1:  # Don't delay after the last token
                time.sleep(delay)

        return results

    def save_raw_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save raw results without statistics - primary data save"""

        with open(output_path.replace('.json', '_dump.txt'), 'w', encoding='utf-8') as f:
            f.write(str(results))

        raw_output = {
            "metadata": {
                "total_tokens": len(results),
                "categories": ["location", "characters", "emotions", "time"],
                "processing_date": datetime.now().isoformat(),
                "description": "Context-aware semantic token labeling using Anthropic API with probabilistic label generation",
                "max_context_tokens": self.max_context_tokens
            },
            "tokens": results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(raw_output, f, indent=2, ensure_ascii=False)

        print(f"\nRaw results saved to: {output_path}")
        print(f"Total tokens processed: {len(results)}")
    def generate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics from the results"""
        stats = {
            'location': {'count': 0, 'unique_values': set()},
            'characters': {'count': 0, 'unique_values': set()},
            'emotions': {'count': 0, 'unique_values': set()},
            'time': {'count': 0, 'unique_values': set()}
        }

        for result in results:
            for category in stats.keys():
                value = result['labels'][category]['value']
                if value != "null":
                    stats[category]['count'] += 1
                    stats[category]['unique_values'].add(value)

        # Convert sets to lists and counts
        final_stats = {}
        for category, data in stats.items():
            final_stats[category] = {
                'count': data['count'],
                'unique_count': len(data['unique_values']),
                'unique_values': list(data['unique_values'])
            }

        return final_stats

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON file - always saves raw data first"""
        # ALWAYS save raw results first - this is the primary output
        self.save_raw_results(results, output_path)

        # Optionally try to generate and add statistics
        try:
            print("\nGenerating statistics...")
            stats = self.generate_statistics(results)

            # Read the raw file we just saved and add statistics
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data["metadata"]["statistics"] = stats

            # Save again with statistics
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print("Statistics added to results file.")
            print("\nStatistics:")
            for category, data in stats.items():
                percentage = (data['count'] / len(results)) * 100
                print(f"- {category.capitalize()}: {data['count']} tokens ({percentage:.1f}%), {data['unique_count']} unique values")

        except Exception as e:
            print(f"\nNote: Could not generate statistics ({e}), but raw data is safely saved.")


def main():
    parser = argparse.ArgumentParser(description='Label tokens using Anthropic API')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--api-key', help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--max-context', type=int, default=500, help='Maximum previous tokens to include as context (default: 500)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls in seconds (default: 0.5)')

    args = parser.parse_args()

    # Get API key
    load_dotenv("./.env")
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please provide API key via --api-key argument or ANTHROPIC_API_KEY environment variable")
        return

    try:
        # Read CSV file
        print(f"Reading {args.input_file}...")
        df = pd.read_csv(args.input_file)

        # Initialize labeler
        labeler = TokenLabeler(api_key, max_context_tokens=args.max_context)

        # Clean tokens
        tokens = labeler.clean_tokens(df)

        if not tokens:
            print("Error: No valid tokens found in the input file")
            return

        # Process tokens
        results = labeler.process_tokens(tokens, delay=args.delay)

        # Save results
        labeler.save_results(results, args.output_file)

        print(f"\nProcessing complete!")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()


# Example usage:
# python token_labeler.py tokens.csv output.json --api-key your_api_key_here
#
# Or with environment variable:
# export ANTHROPIC_API_KEY=your_api_key_here
# python token_labeler.py tokens.csv output.json --max-context 800 --delay 0.3