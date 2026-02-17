# spar-team-recon
Shared mono-repo for our spar project. Down the line, you may want to create project-specific repos.


## Repo structure:
* `shared` - shared code that projects can depend on
* `projects` - folders of code that don't have cross-folder dependencies. Feel free to put unpolished stuff here

## Python package management
We'll use [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python dependency management.

### Setup

On a new machine, install uv with:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Run the command `uv` to make sure it's installed.

Note: if it's not recognized, try running:
```
export PATH="/root/.local/bin:$PATH"
```

### Usage
To add dependencies to the project (which will affect everyone), use `uv add PACKAGE_NAME`. If you want to add a dependency that won't affect everyone (e.g. for a one-off experiment), just use `uv pip install PACKAGE_NAME`. 

You can then activate the virtual environment using `uv run python ...`, or by doing `source .venv/bin/activate` and running things normally. 

### Other useful tools

1. **tmux** — Terminal multiplexer that keeps sessions/experiment runs alive after you shut your computer or disconnect from the virtual machine. Install with `apt install tmux` on Linux (you might need to run `apt update` first), then run `tmux` to start a session, and `tmux attach-session SESSION_NAME` to reconnect to that session. 
2. **ruff** — Fast Python linter and formatter (replaces flake8/black). Already bundled with uv — run `uvx ruff check .` to lint or `uvx ruff format .` to format.
3. **Claude Code** — AI coding assistant in your terminal. Install with `npm install -g @anthropic-ai/claude-code`, then run `claude` in the terminal.

   If the installation isn't working, you may need to install Node.js first:
   ```bash
   # Install nvm (Node version manager)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

   # Load nvm into your current shell
   export NVM_DIR="$HOME/.nvm"
   [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
   [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

   # Install Node.js and Claude Code
   nvm install --lts
   npm install -g @anthropic-ai/claude-code
   ```


