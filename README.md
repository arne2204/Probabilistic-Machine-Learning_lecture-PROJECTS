# Probabilistic-Machine-Learning_lecture-PROJECTS

## Welcome to the repository for the end of semester SoSe2025 projects for the Probabilistic Machine Learning [Lecture](https://github.com/IvaroEkel/Probabilistic-Machine-Learning_Lecture/) by Dr. Alvaro Diaz-Ruelas.

## Some freely available and interesting datasets/sources:

    - Statistisches Bundesamt: [link](https://www.destatis.de/DE/Home/_inhalt.html)
    - 10xgenomics: [lnik](https://www.10xgenomics.com/datasets?configure%5BhitsPerPage%5D=50&configure%5BmaxValuesPerFacet%5D=1000)
    - Yahoo Finance
    - Detailed weather datasets: https://guides.lib.berkeley.edu/climate/data
    


## IMPORTANT: 

- Every time you work on your project locally, start with a pull request: `git pull`, so you do not have conflicts.

- Please include an updated requirements.txt file so I can easily reproduce your environment. 

- Please upload the file(s) with the data you want to analyze to your project folder. But only if the size of the file does not exceed a few MB (~3MB or so). Otherwise, please upload it to google drive or other cloud service. 
    You need to start writing the very first parts of your project, that might need hours of work, such as the data thorough description, cleaning if necessary, and description of your main hypotheses. 


## **How to make your project-ID if you do not have one already?**
Ideally, first send me a project proposal in an email (preferably containing (full or sample) and/or describing the dataset(s)). If I approve the project, I will create the ID for you.
Otherwise:

    1. The first 2 digits of the ID correspond to the day of the month when the project is chosen
    2. Then a hypen: -
    3. Then the order of arrival
    4. Then the initials (starting by surname, then first name) for each member of the group, up to 3 members, and XX for each empty member.

For instance, if I, Alvaro Diaz-Ruelas chooses his project on the 12th of Mai, and I notice that there are already 6 other projects chosen on that day before me, my ID is: 

    12-7DAXXXX.

If, instead of being alone I make the project with a colleague, say, Geoffrey Hinton, then the ID of our project is: 

    12-7DAHGXX.

**How to name the folder of my project?** We append to the project-ID a short string without spaces as descriptive as possible: 12-7DAHGXX_double_descent
Now we can proceed to create a folder under the projects/ folder.

If you created your own project folder (after having my feedback) then, you can upload it with the proper name, only when if it contains a file. Then just create a temp.txt file inside as a placeholder. 
    
## Summarized Repository Structure
- `projects/` – Each student/team will have a folder here.
- `templates/` – Example notebooks or report templates.

## Workflow for Students
1. Fork this repository to your personal GitHub account.
2. Clone your fork and work locally.
3. Create a folder under `projects/` with your assigned PROJECT-ID, if it does not exist already.
4. Commit your work regularly **to your own fork** (let's say that regularly is every working session).
5. When ready, create a Pull Request (PR) to this central repository.
6. I (Alvaro) will review and merge.

You are not allowed to push directly to this repository.

## Rules
- Respect the folder structure.
- Do not modify other project's folders.
- Follow the coding guidelines described in `CONTRIBUTING.md`.


# Repository Tree Structure

    Probabilistic-Machine-Learning_lecture-PROJECTS/
    ├── README.md
    ├── CONTRIBUTING.md
    ├── LICENSE
    ├── projects/
    │   ├── project-id-1/
    │   │   ├── notebooks/
    │   │   ├── data/
    │   │   └── results/
    │   ├── project-id-2/
    │   │   ├── notebooks/
    │   │   ├── data/
    │   │   └── results/
    │   └── ...
    ├── templates/
    │   └── project_report_template.ipynb
    └── .github/
        └── PULL_REQUEST_TEMPLATE.md


## License
This course repository is for educational use only. All rights reserved.
