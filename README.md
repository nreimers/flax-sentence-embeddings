# flax-sentence-embeddings

This repository will be used to share code for the Flax / JAX community event to train sentence embeddings on 1B+ training pairs.

You can add your code by creating a pull request.


## Installation

### Poetry

A Poetry toml is provided to manage dependencies in a virtualenv. Check https://python-poetry.org/

Once you've installed poetry, you can connect to virtual env and update dependencies:
 
```
poetry shell
poetry update
poetry install
```

### requirements.txt

Someone on your platform should generate it once with following command.

```
poetry export -f requirements.txt --output requirements.txt
```

### Rust compiler for hugginface tokenizers

- Hugginface tokenizers require a Rust compiler so install one.

### custom libs

- If you want a specific version of any library, edit the pyproject.toml, add it and/or replace "*" by it.




