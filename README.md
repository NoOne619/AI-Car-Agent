name: README Workflow

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  check-readme:
    runs-on: ubuntu-latest

    steps:
      # Check out the repository
      - name: Checkout code
        uses: actions/checkout@v3

      # Set up Python (since your project uses Python)
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.8"

      # Install dependencies (if needed for README processing)
      - name: Install dependencies
        run: |
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          fi

      # Check if README.md exists
      - name: Check README existence
        run: |
          if [ ! -f README.md ]; then
            echo "Error: README.md not found!"
            exit 1
          fi

      # Optional: Validate README format (e.g., check for key sections)
      - name: Validate README content
        run: |
          grep -q "^#.*AI Car TORCS Agent" README.md || { echo "Error: README missing project title!"; exit 1; }
          grep -q "^##.*Overview" README.md || { echo "Error: README missing Overview section!"; exit 1; }

      # Optional: Generate README (uncomment and customize if you have a generation script)
      # - name: Generate README
      #   run: |
      #     python generate_readme.py  # Path to your README generation script
      #     git config user.name "GitHub Actions"
      #     git config user.email" "actions@github.com"
      #     git add README.md
      #     git commit -m "Auto-generate README" || echo "No changes to commit"
      #     git push origin main

      # Optional: Notify on failure (e.g., Slack, Discord)
      # - name: Notify on failure
      #     if: failure()
      #   uses: slackapi/slack-github-action@v1.2.24
      #   with:
      #     slack-bot-token: ${{ secrets.SLACK_BOT_TOKEN }}
      #     channel-id: 'your-channel-id'
      #     text: 'README check failed for AI Car TORCS Agent project!'
