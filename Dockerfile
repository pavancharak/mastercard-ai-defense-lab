# Serves web-prototype/, the Mastercard Innovation Challenge web prototype.
# Build context is the repo root (mastercard-ai-defense-lab/), not
# web-prototype/ alone: web_prototype/data_sources.py and
# mandate_demo/classifier.py both resolve sibling-pillar paths
# (identify/, generate/, defend/, mandate-demo/output/) relative to
# their own source file location via Path(__file__).resolve().parents[N]
# -- see each file's own path constant. Both packages are installed
# editable (`pip install -e`) specifically so those __file__-relative
# computations keep resolving to the paths copied below, exactly as
# they do in local development, rather than to a site-packages copy.
#
# Only the specific subpaths each pillar's read-only loaders actually
# need at runtime are copied -- not each pillar's own .venv/tests/
# tooling, which would bloat the image and (for .venv) is platform-
# specific anyway.
FROM python:3.12-slim

# xgboost's compiled extension needs libgomp's OpenMP runtime, which
# isn't in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY identify/attack-taxonomy.md identify/attack-taxonomy.md
COPY generate/data/ generate/data/
COPY defend/model/ defend/model/
COPY defend/results/ defend/results/
COPY mandate-demo/pyproject.toml mandate-demo/pyproject.toml
COPY mandate-demo/src/ mandate-demo/src/
COPY mandate-demo/output/ mandate-demo/output/
COPY web-prototype/pyproject.toml web-prototype/pyproject.toml
COPY web-prototype/src/ web-prototype/src/

# NOTE: no .env is copied, deliberately (see .dockerignore) -- in
# production, OPENAI_API_KEY comes from Fly's secrets manager (`fly
# secrets set`), injected directly into the container's real
# environment. Both projects' load_dotenv(..., override=True) calls
# find no .env file here and correctly no-op, leaving that real,
# Fly-managed environment variable as the only source. override=True
# would only matter if a .env WERE present, which it deliberately isn't
# in this image.

RUN pip install --no-cache-dir -e ./mandate-demo \
    && pip install --no-cache-dir -e ./web-prototype

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "web_prototype.app:app", "--app-dir", "web-prototype/src", "--host", "0.0.0.0", "--port", "8080"]
