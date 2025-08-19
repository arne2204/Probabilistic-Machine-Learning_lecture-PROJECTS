from src.pipeline import EnhancedDataPipeline

def main():
    pl = EnhancedDataPipeline()
    pl.run_complete_pipeline("~/projects/Probabilistic-Machine-Learning_lecture-PROJECTS/projects/05-2F_XXXX_health_surveys_mx/project/data/adultos_ensanut2023_w_n.csv")
    
    
if __name__ == "__main__":
    main()