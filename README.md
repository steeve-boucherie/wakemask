# wakemask

> 🚧 **Under construction.** The API and structure are not stable yet.

Estimate wind farm production losses due to turbine downtime (component failures and O&M activities) using farm simulations with [PyWake](https://topfarm.pages.windenergy.dtu.dk/PyWake/).

Classical availability corrections scale the AEP by the effective number of operating turbines. This ignores that a turbine that is down no longer shadows the turbines behind it, so static scaling tends to be pessimistic. `wakemask` quantifies this wake-relief effect by running each downtime scenario as a boolean availability mask over the turbines, with the long-term goal of building a surrogate model for fast O&M optimisation.

## Folder structure

```
wakemask/
├── src/
│   └── wakemask/       # Package source
├── tests/
├── docs/
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.10.

Clone the repository together with its submodules:

```bash
git clone https://github.com/<your-user>/wakemask.git
cd wakemask
```

Install the package and its dependencies:

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra dev       # ruff, pytest, pre-commit
uv sync --extra doc       # documentation tools
uv sync --extra jupyter   # ipykernel
uv sync --all-extras      # everything
```

## License

MIT. See [LICENSE](LICENSE).
