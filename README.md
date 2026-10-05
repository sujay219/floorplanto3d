# FloorPlanTo3D

Modern Python library for converting floor plan images to 3D geometry.

## Installation

```bash
pip install -e .
```

For API dependencies:
```bash
pip install -e ".[api]"
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### As a Library

```python
from floorplanto3d import FloorPlanProcessor

processor = FloorPlanProcessor()
floor_plan = processor.process("floorplan.png")

# Get JSON output
json_output = floor_plan.to_json()
```

### As a Service

```bash
uvicorn floorplanto3d.api.app:app --host 0.0.0.0 --port 8000
```

Then POST an image:
```bash
curl -X POST -F "image=@floorplan.png" http://localhost:8000/process-floorplan
```

## API

- `GET /health` - Health check
- `POST /process-floorplan` - Process floor plan image