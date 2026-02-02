# S37 - cicd-integrations | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S37
  name: cicd-integrations
  title: "CI/CD Integrations"
  wave: W4-Growth
  priority: P2-MEDIUM
  type: implementation

objective: "GitHub Action e GitLab CI templates"

dependencies:
  - S24  # OSS Launch

deliverables:
  - human-layer-action/action.yml
  - human-layer-action/index.js
  - templates/gitlab-ci.yml
  - templates/pre-commit-config.yaml
```

---

## CI/CD INTEGRATIONS

```yaml
integrations:
  github_action:
    name: "human-layer-action"
    marketplace: "GitHub Marketplace"
    usage: |
      - uses: humangr/human-layer-action@v1
        with:
          api-key: ${{ secrets.ANTHROPIC_API_KEY }}

  gitlab_ci:
    name: "human-layer template"
    usage: |
      include:
        - remote: 'https://raw.githubusercontent.com/humangr/human-layer/main/templates/gitlab-ci.yml'

  pre_commit:
    name: "pre-commit hook"
    usage: |
      - repo: https://github.com/humangr/human-layer
        hooks:
          - id: human-layer
```

---

## IMPLEMENTATION SPEC

### GitHub Action (action.yml)

```yaml
name: 'Human Layer Validation'
description: '7 Layers of Human Judgment for AI Agent Validation'
author: 'HumanGR'

inputs:
  api-key:
    description: 'LLM API key (Claude, OpenAI, etc)'
    required: true
  provider:
    description: 'LLM provider'
    required: false
    default: 'claude'
  layers:
    description: 'Layers to run (comma-separated)'
    required: false
    default: 'all'

outputs:
  decision:
    description: 'Validation decision'
  veto-level:
    description: 'Highest veto level'
  findings-count:
    description: 'Number of findings'

runs:
  using: 'composite'
  steps:
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install Human Layer
      shell: bash
      run: pip install human-layer

    - name: Run Validation
      shell: bash
      env:
        ANTHROPIC_API_KEY: ${{ inputs.api-key }}
      run: |
        human-layer validate --diff --format json > result.json

    - name: Parse Results
      shell: bash
      run: |
        echo "decision=$(jq -r '.decision' result.json)" >> $GITHUB_OUTPUT
        echo "veto-level=$(jq -r '.veto_level' result.json)" >> $GITHUB_OUTPUT
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls human-layer-action/action.yml
      ls templates/gitlab-ci.yml

  G1_ACTION_VALID:
    validation: |
      python -c "
      import yaml
      with open('human-layer-action/action.yml') as f:
          action = yaml.safe_load(f)
      assert 'inputs' in action
      assert 'runs' in action
      "
```

---

## REFERÊNCIA

- `./S24_CONTEXT.md` - OSS Launch
- `./S38_CONTEXT.md` - Cockpit Visual
