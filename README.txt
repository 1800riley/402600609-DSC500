# Book Market Analytics Project

## Files
- `Book_Market_Analytics.ipynb`: complete end-to-end notebook.
- `requirements.txt`: packages needed to run the notebook.
- `books_raw_template.csv`: empty CSV with raw-data column headers; the notebook creates the real dataset when run.
- `Report_Guide.docx`: report outline and submission guidance.

## How to run
1. Install Python and Jupyter Notebook or use Google Colab.
2. Open `Book_Market_Analytics.ipynb`.
3. Run the cells from top to bottom while connected to the internet.
4. Wait for the scraper to finish. It visits catalogue listing pages and book detail pages, so it may take several minutes.
5. Check the collected row count and any failed pages.
6. The notebook creates a `book_market_outputs` folder containing raw and cleaned CSV files, visualisations, model predictions and a report populated with actual results.

## Important
The supplied CSV template is not a scraped dataset. Do not submit it as if it contains real observations. The real CSV files are created when the notebook runs successfully. The site is a practice website with randomly assigned prices and ratings, so conclusions must be limited to this dataset.

## GitHub Repository
link: https://github.com/1800riley/402600609-DSC500
