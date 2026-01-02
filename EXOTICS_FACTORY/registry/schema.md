# EXOTICS_FACTORY Spec Schema

This schema describes the expected structure for each `spec.yaml` under
`EXOTICS_FACTORY/families/<family_id>/spec.yaml`.

## Top-level keys

- `schema_version` (string, required)
- `family` (object, required)
- `sources` (list, required)
- `channels` (list, required)
- `model` (object, required)
- `outputs` (object, required)
- `runtime` (object, required)

## `family` object

- `id` (string, required)
- `name` (string, required)
- `category` (string, required; `exotic` or `control`)
- `states` (list of strings, required)
- `channels` (list of strings, required)
- `preferred_backends` (list of strings, required)
- `model_class` (string, required)
- `amplitude_level_requires` (list of strings, required)
- `proxy_only` (boolean, required)
- `source_pointers` (list of strings, required)

## `sources` list

Each entry:

- `id` (string, required)
- `backend` (string, required)
- `description` (string, required)
- `placeholders` (list of strings, required)

## `channels` list

Each entry:

- `id` (string, required)
- `label` (string, required)
- `source_ref` (string, required; must match a source `id`)
- `notes` (string, optional)

## `model` object

- `type` (string, required)
- `parameters` (list of strings, required)
- `shared_structure` (string, required)
- `notes` (string, optional)

## `outputs` object

- `output_dir` (string, required)
- `report_template` (string, required)
- `artifacts` (list of strings, required)

## `runtime` object

- `dry_run_only` (boolean, required)
- `steps` (list of strings, required)
- `backend_settings` (object, required)
