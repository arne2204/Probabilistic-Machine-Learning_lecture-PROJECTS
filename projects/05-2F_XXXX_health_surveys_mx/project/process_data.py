from src.pipeline import EnhancedDataPipeline
import pandas as pd

def main():
    pl = EnhancedDataPipeline()
    df = pl.run_complete_pipeline("~/projects/Probabilistic-Machine-Learning_lecture-PROJECTS/projects/05-2F_XXXX_health_surveys_mx/project/data/adultos_ensanut2023_w_n.csv")
    print(df['hora_ini_2'].unique())
    
    
    
if __name__ == "__main__":
    main()