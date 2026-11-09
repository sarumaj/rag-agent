# Changelog

All notable changes to this project are documented in this file. The release bot
writes each version's section from the commits (Conventional Commits) and the hand-written
Unreleased section.

## 0.3.2 - 2026-11-09

### Dependencies

- update all patch dependencies
- a consistent langchain 0.3 set: langchain-core 0.3.86 and langchain-text-splitters 0.3.11, which langchain 0.3.30 needs, and langchain-community 0.3.31 (0.4 needs langchain-core 1.x); pydantic-settings 2.10.1 and pytest 8.4.2 for them and pytest-asyncio; langchain-huggingface[full], whose 0.3 moved sentence-transformers into that extra

## 0.3.1 - 2026-11-09

### Dependencies

- update dependency unstructured to v0.24.0 [SECURITY]
- update dependency aiohttp to v3.14.3 [SECURITY]
- update dependency langchain-community to v0.3.27 [SECURITY] (#81)
- update dependency langchain to v0.3.30 [SECURITY] (#96)
- update pypa/gh-action-pypi-publish action to v1.13.0 [SECURITY] (#84)

