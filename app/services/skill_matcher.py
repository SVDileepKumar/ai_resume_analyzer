"""Skill extraction, implication expansion, cosine-fuzzy matching, and role profiles."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

logger = logging.getLogger(__name__)
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(filename: str) -> Any:
    """Load JSON from app/data/. On failure log clearly and re-raise so startup fails fast."""
    path = _DATA_DIR / filename
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.error("Data file not found: %s", path)
        raise FileNotFoundError(f"Skill data file not found: {path}. Ensure {filename} exists in app/data/.") from None
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", path, e)
        raise ValueError(f"Invalid JSON in {filename}: {e}") from e
    except OSError as e:
        logger.error("Cannot read data file %s: %s", path, e)
        raise RuntimeError(f"Cannot read {filename}: {e}. Check permissions and path.") from e


# ---------------------------------------------------------------------------
# Data loaded once at import time (validated; clear error on failure)
# ---------------------------------------------------------------------------
try:
    _RAW_SKILL_DB: dict[str, list[str]] = _load_json("skill_db.json")
    ROLE_PROFILES: dict[str, dict] = _load_json("role_profiles.json")
except (FileNotFoundError, ValueError, OSError, RuntimeError):
    raise  # Re-raise with clear message so startup fails fast

# Flatten all known skills into a lookup set
ALL_KNOWN_SKILLS: set[str] = {
    skill.strip().lower()
    for category_skills in _RAW_SKILL_DB.values()
    for skill in category_skills
}

# ---------------------------------------------------------------------------
# Skill Implication Map
# DEPRECATED (2026-04-09): The implication graph, alias tables, and skill families are
# no longer part of the active scoring path. They are retained here solely so the
# fallback match_skills() function (called when LLM extraction fails) continues to work.
# New skill inference is handled by extract_skills_llm() in llm_service.py.
# If someone knows skill X, they implicitly know skills Y, Z, ...
# This models real-world knowledge transfer between related technologies.
# ---------------------------------------------------------------------------
SKILL_IMPLIES: dict[str, list[str]] = {
    # ---------------------------------------------------------------
    # Languages: advanced implies base
    # ---------------------------------------------------------------
    "c++":          ["c"],
    "c#":           ["c"],
    "typescript":   ["javascript"],
    "kotlin":       ["java"],
    "scala":        ["java"],
    "objective-c":  ["c"],
    "dart":         ["javascript"],
    "groovy":       ["java"],
    "elixir":       ["ruby"],
    "rust":         ["c", "c++"],
    "swift":        ["objective-c"],

    # ---------------------------------------------------------------
    # Frontend frameworks → core web + JS
    # ---------------------------------------------------------------
    "react":        ["javascript", "html", "css", "typescript"],
    "angular":      ["javascript", "typescript", "html", "css"],
    "vue":          ["javascript", "html", "css"],
    "svelte":       ["javascript", "html", "css"],
    "next.js":      ["react", "javascript", "html", "css", "typescript"],
    "nuxt.js":      ["vue", "javascript", "html", "css"],
    "gatsby":       ["react", "javascript"],
    "react native": ["react", "javascript", "mobile development"],
    "htmx":         ["html", "javascript"],
    "alpine.js":    ["javascript", "html"],
    "storybook":    ["react", "javascript"],

    # ---------------------------------------------------------------
    # Backend frameworks → language
    # ---------------------------------------------------------------
    "django":       ["python", "rest api"],
    "flask":        ["python"],
    "fastapi":      ["python", "rest api"],
    "spring boot":  ["java", "spring", "rest api"],
    "spring":       ["java"],
    "express":      ["node.js", "javascript", "rest api"],
    "node.js":      ["javascript"],
    "rails":        ["ruby", "rest api"],
    "laravel":      ["php", "rest api"],
    "asp.net":      [".net", "c#", "rest api"],
    ".net":         ["c#"],
    "graphql":      ["rest api"],
    "grpc":         ["rest api"],
    "celery":       ["python"],
    "rabbitmq":     ["microservices"],
    "kafka":        ["microservices", "streaming"],

    # ---------------------------------------------------------------
    # DevOps: orchestration → containerization → infra
    # ---------------------------------------------------------------
    "kubernetes":   ["docker", "linux"],
    "helm":         ["kubernetes", "docker"],
    "ecs":          ["docker", "aws"],
    "eks":          ["kubernetes", "docker", "aws"],
    "terraform":    ["infrastructure as code", "cloud"],
    "ansible":      ["infrastructure as code", "linux"],
    "prometheus":   ["monitoring"],
    "grafana":      ["monitoring", "data visualization"],
    "datadog":      ["monitoring"],
    "splunk":       ["monitoring"],
    "nginx":        ["linux"],
    "docker":       ["linux"],

    # ---------------------------------------------------------------
    # Cloud services → platform
    # ---------------------------------------------------------------
    "lambda":       ["aws", "serverless"],
    "s3":           ["aws"],
    "ec2":          ["aws"],
    "dynamodb":     ["aws", "nosql"],
    "cloud functions": ["gcp", "serverless"],
    "cloud run":    ["gcp", "docker"],
    "app engine":   ["gcp"],
    "bigquery":     ["gcp", "sql", "data warehouse"],
    "azure":        ["cloud"],
    "aws":          ["cloud"],
    "gcp":          ["cloud"],
    "google cloud": ["cloud"],
    "heroku":       ["cloud"],
    "vercel":       ["cloud"],
    "netlify":      ["cloud"],
    "cloudflare":   ["cloud"],
    "digitalocean": ["cloud"],

    # ---------------------------------------------------------------
    # Data science / ML: frameworks → language + discipline
    # ---------------------------------------------------------------
    "pandas":       ["python", "data analysis"],
    "numpy":        ["python"],
    "scipy":        ["python", "statistics"],
    "scikit-learn": ["python", "machine learning", "statistics"],
    "tensorflow":   ["python", "deep learning", "machine learning", "neural networks"],
    "pytorch":      ["python", "deep learning", "machine learning", "neural networks"],
    "keras":        ["python", "deep learning", "neural networks"],
    "xgboost":      ["python", "machine learning"],
    "lightgbm":     ["python", "machine learning"],
    "hugging face": ["python", "nlp", "deep learning", "machine learning"],
    "spacy":        ["python", "nlp"],
    "nltk":         ["python", "nlp"],
    "opencv":       ["python", "computer vision"],
    "mlops":        ["machine learning", "docker", "ci/cd"],
    "deep learning": ["machine learning", "neural networks"],
    "nlp":          ["machine learning"],
    "computer vision": ["machine learning", "deep learning"],

    # ---------------------------------------------------------------
    # Data engineering
    # ---------------------------------------------------------------
    "airflow":      ["python", "etl", "data pipeline"],
    "dbt":          ["sql", "data warehouse", "data modeling"],
    "spark":        ["python", "sql", "data pipeline", "batch processing"],
    "databricks":   ["spark", "python", "data lake"],
    "snowflake":    ["sql", "data warehouse"],
    "hadoop":       ["data pipeline", "batch processing"],
    "hive":         ["sql", "hadoop"],
    "etl":          ["data pipeline"],
    "data warehouse": ["data modeling", "sql"],
    "data lake":    ["data warehouse"],
    "streaming":    ["data pipeline"],

    # ---------------------------------------------------------------
    # Testing frameworks → language + practice
    # ---------------------------------------------------------------
    "pytest":       ["python", "unit testing"],
    "jest":         ["javascript", "unit testing"],
    "junit":        ["java", "unit testing"],
    "testng":       ["java", "unit testing"],
    "cypress":      ["javascript", "end-to-end testing", "automation testing"],
    "playwright":   ["javascript", "end-to-end testing", "automation testing"],
    "selenium":     ["automation testing"],
    "cucumber":     ["bdd", "automation testing"],
    "jmeter":       ["load testing", "performance testing"],
    "k6":           ["load testing", "performance testing"],
    "postman":      ["api testing"],
    "test driven development": ["unit testing"],

    # ---------------------------------------------------------------
    # Mobile → platform knowledge
    # ---------------------------------------------------------------
    "flutter":      ["dart", "mobile development", "ios", "android"],
    "swiftui":      ["swift", "ios", "mobile development"],
    "jetpack compose": ["kotlin", "android", "mobile development"],
    "xamarin":      ["c#", ".net", "mobile development", "ios", "android"],
    "ios":          ["mobile development"],
    "android":      ["mobile development"],

    # ---------------------------------------------------------------
    # CSS tools → CSS
    # ---------------------------------------------------------------
    "tailwind":     ["css", "responsive design"],
    "bootstrap":    ["css", "responsive design"],
    "sass":         ["css"],
    "less":         ["css"],
    "material ui":  ["css", "react"],

    # ---------------------------------------------------------------
    # State management → framework
    # ---------------------------------------------------------------
    "redux":        ["react", "javascript"],
    "zustand":      ["react", "javascript"],

    # ---------------------------------------------------------------
    # CI/CD tools → CI/CD + VCS
    # ---------------------------------------------------------------
    "github actions": ["ci/cd", "git"],
    "gitlab ci":    ["ci/cd", "git"],
    "jenkins":      ["ci/cd"],
    "circleci":     ["ci/cd"],
    "travis ci":    ["ci/cd"],

    # ---------------------------------------------------------------
    # Databases / SQL variants → SQL
    # ---------------------------------------------------------------
    "mysql":        ["sql", "schema design"],
    "postgresql":   ["sql", "schema design"],
    "sqlite":       ["sql"],
    "oracle":       ["sql", "schema design"],
    "sql server":   ["sql", "schema design"],
    "mariadb":      ["sql", "schema design"],
    "plsql":        ["sql"],
    "cockroachdb":  ["sql", "schema design"],
    "mongodb":      ["nosql"],
    "cassandra":    ["nosql"],
    "couchdb":      ["nosql"],
    "redis":        ["nosql", "caching"],
    "neo4j":        ["nosql"],
    "elasticsearch": ["nosql"],
    "firebase":     ["nosql", "gcp"],
    "supabase":     ["postgresql", "sql"],
    "influxdb":     ["nosql"],

    # ---------------------------------------------------------------
    # Security → base knowledge
    # ---------------------------------------------------------------
    "penetration testing": ["cybersecurity"],
    "owasp":        ["cybersecurity"],
    "vulnerability assessment": ["cybersecurity"],
    "security audit": ["cybersecurity", "compliance"],
    "soc 2":        ["compliance"],
    "gdpr":         ["compliance"],
    "encryption":   ["cybersecurity"],

    # ---------------------------------------------------------------
    # Analytics / Visualization
    # ---------------------------------------------------------------
    "tableau":      ["data visualization", "data analysis"],
    "power bi":     ["data visualization", "data analysis"],
    "looker":       ["data visualization", "sql"],
    "matplotlib":   ["python", "data visualization"],
    "seaborn":      ["python", "data visualization", "matplotlib"],
    "plotly":       ["python", "data visualization"],
    "a/b testing":  ["statistics", "data analysis"],
    "regression analysis": ["statistics"],
    "hypothesis testing": ["statistics"],

    # ---------------------------------------------------------------
    # Enterprise platforms → domain
    # ---------------------------------------------------------------
    "apex":         ["salesforce"],
    "mulesoft":     ["microservices"],
    "informatica":  ["etl", "data pipeline"],
    "outsystems":   ["rest api"],
    "successfactors": ["sap"],

    # ---------------------------------------------------------------
    # AI / GenAI / LLM ecosystem
    # ---------------------------------------------------------------
    "langchain":    ["python", "llm", "generative ai", "rag"],
    "llm":          ["generative ai", "machine learning"],
    "rag":          ["llm", "generative ai"],
    "fine tuning":  ["machine learning", "deep learning"],
    "prompt engineering": ["llm", "generative ai"],
    "generative ai": ["machine learning"],

    # ---------------------------------------------------------------
    # Methodologies → related practices
    # ---------------------------------------------------------------
    "scrum":        ["agile"],
    "kanban":       ["agile"],
    "lean":         ["agile"],
    "domain driven design": ["clean architecture", "design patterns"],
    "event driven": ["microservices"],
    "cqrs":         ["event driven", "design patterns"],

    # ---------------------------------------------------------------
    # Additional DevOps / Infrastructure
    # ---------------------------------------------------------------
    "argocd":       ["kubernetes", "ci/cd", "docker"],
    "istio":        ["kubernetes", "service mesh"],
    "pulumi":       ["infrastructure as code", "cloud"],
    "vault":        ["cybersecurity"],
    "consul":       ["microservices"],
    "packer":       ["infrastructure as code"],

    # ---------------------------------------------------------------
    # Additional cloud / AWS / Azure
    # ---------------------------------------------------------------
    "sqs":          ["aws", "microservices"],
    "sns":          ["aws"],
    "kinesis":      ["aws", "streaming"],
    "step functions": ["aws", "serverless"],
    "cloudformation": ["aws", "infrastructure as code"],
    "azure devops": ["azure", "ci/cd"],
    "azure functions": ["azure", "serverless"],
    "cosmos db":    ["azure", "nosql"],

    # ---------------------------------------------------------------
    # Additional frontend frameworks / tools
    # ---------------------------------------------------------------
    "remix":        ["react", "javascript", "typescript", "html", "css"],
    "astro":        ["javascript", "html", "css"],
    "solid.js":     ["javascript", "html", "css"],
    "qwik":         ["javascript", "typescript", "html", "css"],
    "lit":          ["javascript", "web components", "html", "css"],
    "web components": ["javascript", "html"],
    "pwa":          ["javascript", "html", "service worker"],
    "emotion":      ["css", "react"],
    "styled components": ["css", "react"],
    "chakra ui":    ["css", "react", "responsive design"],
    "ant design":   ["css", "react"],
    "radix ui":     ["react"],
    "shadcn":       ["react", "tailwind", "css"],
    "turbopack":    ["javascript"],
    "esbuild":      ["javascript"],
    "rollup":       ["javascript"],
    "parcel":       ["javascript"],
    "babel":        ["javascript"],
    "react query":  ["react", "javascript"],
    "tanstack":     ["react", "javascript"],
    "mobx":         ["react", "javascript"],
    "recoil":       ["react", "javascript"],
    "jotai":        ["react", "javascript"],
    "pinia":        ["vue", "javascript"],
    "three.js":     ["javascript", "webgl"],
    "d3.js":        ["javascript", "data visualization"],
    "gsap":         ["javascript", "css"],
    "framer motion": ["react", "javascript"],
    "wasm":         ["webassembly"],

    # ---------------------------------------------------------------
    # Additional backend frameworks
    # ---------------------------------------------------------------
    "nestjs":       ["node.js", "typescript", "javascript", "rest api"],
    "koa":          ["node.js", "javascript"],
    "hapi":         ["node.js", "javascript"],
    "fastify":      ["node.js", "javascript", "rest api"],
    "gin":          ["go", "rest api"],
    "echo":         ["go", "rest api"],
    "fiber":        ["go", "rest api"],
    "actix":        ["rust", "rest api"],
    "rocket":       ["rust"],
    "axum":         ["rust", "rest api"],
    "phoenix":      ["elixir", "rest api"],
    "ktor":         ["kotlin", "rest api"],
    "quarkus":      ["java", "rest api", "microservices"],
    "micronaut":    ["java", "rest api", "microservices"],
    "dropwizard":   ["java", "rest api"],
    "vertx":        ["java"],
    "trpc":         ["typescript", "rest api"],
    "protobuf":     ["grpc"],
    "sidekiq":      ["ruby", "redis"],
    "bull":         ["node.js", "redis"],
    "event sourcing": ["event driven", "microservices"],
    "saga pattern": ["microservices", "event driven"],

    # ---------------------------------------------------------------
    # Additional databases / ORM
    # ---------------------------------------------------------------
    "timescaledb":  ["postgresql", "sql", "time series"],
    "clickhouse":   ["sql", "data warehouse"],
    "duckdb":       ["sql", "data analysis"],
    "vitess":       ["mysql", "sql"],
    "planetscale":  ["mysql", "sql"],
    "memcached":    ["caching"],
    "etcd":         ["kubernetes"],
    "milvus":       ["vector database", "machine learning"],
    "pinecone":     ["vector database", "machine learning"],
    "weaviate":     ["vector database", "machine learning"],
    "qdrant":       ["vector database", "machine learning"],
    "chroma":       ["vector database", "machine learning"],
    "prisma":       ["sql", "typescript"],
    "drizzle":      ["sql", "typescript"],
    "sequelize":    ["sql", "node.js", "javascript"],
    "typeorm":      ["sql", "typescript"],
    "sqlalchemy":   ["sql", "python"],
    "hibernate":    ["sql", "java"],
    "active record": ["sql", "ruby"],

    # ---------------------------------------------------------------
    # Additional cloud services
    # ---------------------------------------------------------------
    "rds":          ["aws", "sql"],
    "aurora":       ["aws", "sql", "postgresql"],
    "redshift":     ["aws", "sql", "data warehouse"],
    "elasticache":  ["aws", "redis", "caching"],
    "cloudfront":   ["aws", "cdn"],
    "route 53":     ["aws", "dns"],
    "fargate":      ["aws", "docker", "serverless"],
    "ecr":          ["aws", "docker"],
    "glue":         ["aws", "etl", "data pipeline"],
    "athena":       ["aws", "sql"],
    "emr":          ["aws", "spark", "hadoop"],
    "sagemaker":    ["aws", "machine learning", "mlops"],
    "bedrock":      ["aws", "llm", "generative ai"],
    "azure aks":    ["azure", "kubernetes"],
    "azure blob":   ["azure"],
    "azure sql":    ["azure", "sql", "schema design"],
    "azure ad":     ["azure", "iam"],
    "gke":          ["gcp", "kubernetes"],
    "cloud storage": ["gcp"],
    "cloud sql":    ["gcp", "sql"],
    "pub/sub":      ["gcp", "streaming", "microservices"],
    "vertex ai":    ["gcp", "machine learning", "mlops"],
    "cdk":          ["aws", "infrastructure as code"],
    "sam":          ["aws", "serverless", "infrastructure as code"],
    "fly.io":       ["cloud"],
    "railway":      ["cloud"],
    "render":       ["cloud"],

    # ---------------------------------------------------------------
    # Additional DevOps / Observability
    # ---------------------------------------------------------------
    "fluxcd":       ["kubernetes", "ci/cd", "gitops"],
    "tekton":       ["kubernetes", "ci/cd"],
    "spinnaker":    ["ci/cd"],
    "harness":      ["ci/cd"],
    "buildkite":    ["ci/cd"],
    "drone":        ["ci/cd", "docker"],
    "bamboo":       ["ci/cd"],
    "opentelemetry": ["observability", "monitoring"],
    "jaeger":       ["observability", "monitoring"],
    "zipkin":       ["observability", "monitoring"],
    "elastic apm":  ["observability", "monitoring", "elasticsearch"],
    "new relic":    ["monitoring", "observability"],
    "dynatrace":    ["monitoring", "observability"],
    "pagerduty":    ["monitoring", "incident management"],
    "loki":         ["monitoring", "observability"],
    "fluentd":      ["monitoring", "observability"],
    "logstash":     ["monitoring", "elasticsearch"],
    "envoy":        ["service mesh", "load balancing"],
    "traefik":      ["reverse proxy", "load balancing"],
    "haproxy":      ["load balancing"],
    "caddy":        ["reverse proxy"],
    "podman":       ["docker", "linux"],
    "containerd":   ["docker"],
    "kustomize":    ["kubernetes"],
    "crossplane":   ["kubernetes", "infrastructure as code"],
    "skaffold":     ["kubernetes", "docker"],
    "terragrunt":   ["terraform", "infrastructure as code"],
    "sonarqube":    ["code review"],
    "trivy":        ["container security", "cybersecurity"],
    "snyk":         ["cybersecurity"],
    "checkov":      ["infrastructure as code", "cybersecurity"],
    "gitops":       ["ci/cd", "git"],
    "feature flag": ["ci/cd"],
    "chaos engineering": ["site reliability"],

    # ---------------------------------------------------------------
    # Additional data science / AI
    # ---------------------------------------------------------------
    "llamaindex":   ["python", "llm", "rag", "generative ai"],
    "autogen":      ["python", "llm", "agents", "generative ai"],
    "crewai":       ["python", "llm", "agents", "generative ai"],
    "semantic kernel": ["llm", "generative ai"],
    "openai api":   ["llm", "generative ai"],
    "anthropic api": ["llm", "generative ai"],
    "gemini api":   ["llm", "generative ai"],
    "ollama":       ["llm"],
    "transformers": ["python", "hugging face", "nlp", "deep learning"],
    "diffusers":    ["python", "hugging face", "deep learning"],
    "stable diffusion": ["deep learning", "generative ai", "computer vision"],
    "agents":       ["llm", "generative ai"],
    "function calling": ["llm", "generative ai"],
    "embeddings":   ["machine learning", "nlp"],
    "transfer learning": ["deep learning", "machine learning"],
    "few shot learning": ["machine learning"],
    "model quantization": ["deep learning", "model deployment"],
    "onnx":         ["deep learning", "model deployment"],
    "tensorrt":     ["deep learning", "model deployment"],
    "mlflow":       ["mlops", "machine learning"],
    "wandb":        ["mlops", "machine learning"],
    "kubeflow":     ["mlops", "kubernetes", "machine learning"],
    "metaflow":     ["mlops", "python", "machine learning"],
    "bentoml":      ["model deployment", "python"],
    "catboost":     ["python", "machine learning", "gradient boosting"],
    "random forest": ["machine learning"],
    "svm":          ["machine learning"],
    "logistic regression": ["machine learning", "statistics"],
    "linear regression": ["machine learning", "statistics"],
    "gan":          ["deep learning", "neural networks"],
    "cnn":          ["deep learning", "neural networks"],
    "rnn":          ["deep learning", "neural networks"],
    "lstm":         ["rnn", "deep learning", "neural networks"],
    "bert":         ["nlp", "deep learning", "transformers"],
    "gpt":          ["nlp", "deep learning", "llm"],
    "yolo":         ["computer vision", "deep learning", "object detection"],
    "object detection": ["computer vision", "deep learning"],
    "image segmentation": ["computer vision", "deep learning"],
    "text classification": ["nlp", "machine learning"],
    "sentiment analysis": ["nlp", "machine learning"],
    "named entity recognition": ["nlp", "machine learning"],
    "speech recognition": ["deep learning", "machine learning"],
    "pca":          ["dimensionality reduction", "statistics"],

    # ---------------------------------------------------------------
    # Additional data engineering
    # ---------------------------------------------------------------
    "flink":        ["streaming", "data pipeline", "java"],
    "beam":         ["data pipeline"],
    "nifi":         ["etl", "data pipeline"],
    "prefect":      ["python", "data pipeline", "etl"],
    "dagster":      ["python", "data pipeline", "etl"],
    "mage":         ["python", "data pipeline", "etl"],
    "fivetran":     ["etl", "data pipeline"],
    "airbyte":      ["etl", "data pipeline"],
    "debezium":     ["cdc", "streaming"],
    "delta lake":   ["spark", "data lake"],
    "iceberg":      ["data lake"],
    "data mesh":    ["data governance"],
    "data catalog": ["data governance"],
    "great expectations": ["data quality", "python"],
    "parquet":      ["data pipeline"],
    "cdc":          ["streaming", "data pipeline"],

    # ---------------------------------------------------------------
    # Additional testing
    # ---------------------------------------------------------------
    "mocha":        ["javascript", "unit testing"],
    "chai":         ["javascript", "unit testing"],
    "vitest":       ["javascript", "unit testing"],
    "testing library": ["javascript", "unit testing"],
    "rspec":        ["ruby", "unit testing"],
    "minitest":     ["ruby", "unit testing"],
    "capybara":     ["ruby", "end-to-end testing"],
    "appium":       ["mobile development", "automation testing"],
    "detox":        ["react native", "mobile development", "automation testing"],
    "robot framework": ["automation testing", "python"],
    "karate":       ["api testing", "automation testing"],
    "rest assured": ["api testing", "java"],
    "gatling":      ["load testing", "performance testing", "scala"],
    "locust":       ["load testing", "performance testing", "python"],
    "pact":         ["contract testing", "api testing"],
    "percy":        ["visual regression testing"],
    "chromatic":    ["visual regression testing", "storybook"],
    "allure":       ["test reporting"],

    # ---------------------------------------------------------------
    # Additional tools / package managers
    # ---------------------------------------------------------------
    "nx":           ["monorepo", "javascript"],
    "turborepo":    ["monorepo", "javascript"],
    "lerna":        ["monorepo", "javascript"],
    "bazel":        ["ci/cd"],
    "gradle":       ["java"],
    "maven":        ["java"],
    "sbt":          ["scala"],
    "bun":          ["javascript"],
    "deno":         ["javascript", "typescript"],
    "poetry":       ["python"],
    "uv":           ["python"],
    "conda":        ["python"],

    # ---------------------------------------------------------------
    # Blockchain / Web3
    # ---------------------------------------------------------------
    "ethereum":     ["blockchain", "smart contracts"],
    "solana":       ["blockchain", "smart contracts"],
    "polygon":      ["blockchain", "ethereum"],
    "hardhat":      ["ethereum", "smart contracts", "javascript"],
    "truffle":      ["ethereum", "smart contracts", "javascript"],
    "foundry":      ["ethereum", "smart contracts"],
    "ethers.js":    ["ethereum", "javascript"],
    "web3.js":      ["ethereum", "javascript"],
    "smart contracts": ["blockchain"],
    "defi":         ["blockchain"],
    "nft":          ["blockchain"],
    "web3":         ["blockchain"],

    # ---------------------------------------------------------------
    # Mobile: additional
    # ---------------------------------------------------------------
    "kotlin multiplatform": ["kotlin", "mobile development", "ios", "android"],
    "capacitor":    ["javascript", "mobile development"],
    "ionic":        ["javascript", "mobile development"],
    "expo":         ["react native", "javascript", "mobile development"],
    "fastlane":     ["ios", "android", "ci/cd"],
    "cocoapods":    ["ios", "swift"],
    "realm":        ["mobile development", "nosql"],
    "crashlytics":  ["mobile development", "monitoring"],

    # ---------------------------------------------------------------
    # Security: additional
    # ---------------------------------------------------------------
    "nist":         ["compliance", "cybersecurity"],
    "iso 27001":    ["compliance", "cybersecurity"],
    "oauth 2.0":    ["oauth", "cybersecurity"],
    "openid connect": ["oauth", "cybersecurity", "sso"],
    "sso":          ["cybersecurity"],
    "mfa":          ["cybersecurity"],
    "waf":          ["cybersecurity", "network security"],
    "sast":         ["cybersecurity", "application security"],
    "dast":         ["cybersecurity", "application security"],
    "sca":          ["cybersecurity", "supply chain security"],
    "container security": ["cybersecurity", "docker"],
    "supply chain security": ["cybersecurity"],

    # ---------------------------------------------------------------
    # Enterprise platforms: additional
    # ---------------------------------------------------------------
    "sharepoint":   ["microsoft 365"],
    "power automate": ["microsoft 365", "power platform"],
    "power apps":   ["microsoft 365", "power platform"],
    "okta":         ["iam", "sso", "cybersecurity"],
    "auth0":        ["iam", "sso", "oauth"],
    "twilio":       ["rest api"],
    "stripe":       ["rest api"],
    "contentful":   ["rest api"],
    "sanity":       ["rest api"],
    "strapi":       ["node.js", "rest api"],
    "shopify":      ["rest api"],

    # ---------------------------------------------------------------
    # Embedded / IoT
    # ---------------------------------------------------------------
    "iot":          ["embedded systems"],
    "raspberry pi": ["linux", "python"],
    "arduino":      ["c", "embedded systems"],
    "mqtt":         ["iot", "microservices"],
    "edge computing": ["iot", "cloud"],
    "tinyml":       ["machine learning", "embedded systems"],

    # ---------------------------------------------------------------
    # Game development
    # ---------------------------------------------------------------
    "unity":        ["c#", "game programming"],
    "unreal engine": ["c++", "game programming"],
    "godot":        ["game programming"],

    # ---------------------------------------------------------------
    # Analytics: additional
    # ---------------------------------------------------------------
    "cohort analysis": ["data analysis", "statistics"],
    "funnel analysis": ["data analysis"],
    "churn analysis": ["data analysis", "statistics"],
    "causal inference": ["statistics", "machine learning"],
    "experimental design": ["statistics"],

    # ---------------------------------------------------------------
    # Data viz: additional
    # ---------------------------------------------------------------
    "kibana":       ["elasticsearch", "data visualization"],
    "jupyter notebook": ["python"],
    "streamlit":    ["python", "data visualization"],
    "dash":         ["python", "plotly", "data visualization"],
    "google data studio": ["data visualization", "data analysis"],

    # ---------------------------------------------------------------
    # Methodologies: additional
    # ---------------------------------------------------------------
    "safe":         ["agile", "scrum"],
    "xp":           ["agile"],
    "twelve factor app": ["cloud native", "microservices"],
    "hexagonal architecture": ["clean architecture", "design patterns"],
    "onion architecture": ["clean architecture", "design patterns"],
    "distributed systems": ["system design"],
    "cap theorem":  ["distributed systems"],
}

# ---------------------------------------------------------------------------
# DEPRECATED (2026-04-09): Skill families retained for fallback match_skills() only.
# New skill extraction handles peer equivalences contextually via LLM.
#
# Skill family / peer equivalence map.
#
# When a JD requires skill X and the resume has skill Y where both X and Y
# belong to the same family, the candidate partially satisfies the requirement.
# This handles the "sibling technology" problem:
#   - JD requires "postgresql" → resume has "mysql" → same SQL DB family → partial match
#   - JD requires "terraform"  → resume has "cdk"   → same IaC family → partial match
#   - JD requires "oauth 2.0"  → resume has "jwt"   → same auth family → partial match
#
# Each family maps to a set of peer skill names.  Membership is symmetric:
# any member satisfies any other member of the same family at FAMILY_PARTIAL_CREDIT.
#
# Design rules:
#   1. Families describe same-purpose interchangeable tools, not "related" topics.
#      (kubernetes and docker are related but not peers — you can't swap them)
#   2. Generic super-skills (e.g. "sql") are included so specific → generic is implicit.
#   3. Keep families tight — false positives hurt candidates more than false negatives.
# ---------------------------------------------------------------------------
SKILL_FAMILIES: dict[str, set[str]] = {
    # Relational databases (SQL dialects) — any SQL DB satisfies another SQL DB req at partial credit
    "sql_databases": {
        "sql", "postgresql", "mysql", "sql server", "mariadb", "sqlite",
        "oracle", "cockroachdb", "azure sql", "cloud sql", "aurora",
        "timescaledb", "clickhouse", "vitess", "planetscale",
    },
    # NoSQL databases
    "nosql_databases": {
        "nosql", "mongodb", "dynamodb", "cassandra", "couchdb", "redis",
        "elasticsearch", "neo4j", "influxdb", "cosmos db", "firebase", "supabase",
    },
    # Infrastructure as Code tools
    "iac_tools": {
        "infrastructure as code", "terraform", "cdk", "cloudformation",
        "pulumi", "ansible", "sam", "bicep", "packer", "crossplane", "terragrunt",
    },
    # Authentication / Authorization standards and implementations
    "auth_standards": {
        "oauth", "oauth 2.0", "openid connect", "jwt", "saml", "sso",
        "iam", "auth0", "okta", "azure ad", "cognito", "keycloak",
        "passport", "oidc",
    },
    # Container orchestration
    "container_orchestration": {
        "kubernetes", "docker swarm", "ecs", "eks", "gke", "azure aks",
        "nomad", "mesos", "fargate",
    },
    # Message queues / event streaming
    "message_queue": {
        "kafka", "rabbitmq", "sqs", "sns", "pub/sub", "nats", "activemq",
        "kinesis", "azure service bus", "redis",
    },
    # CI/CD platforms
    "cicd_platforms": {
        "ci/cd", "github actions", "gitlab ci", "jenkins", "circleci",
        "travis ci", "azure devops", "buildkite", "bamboo", "harness",
        "spinnaker", "teamcity", "drone", "argocd", "fluxcd", "tekton",
    },
    # Monitoring / Observability
    "monitoring_tools": {
        "monitoring", "observability", "prometheus", "grafana", "datadog",
        "new relic", "dynatrace", "splunk", "elastic apm", "opentelemetry",
        "jaeger", "zipkin", "cloudwatch",
    },
    # Data warehouse platforms
    "data_warehouses": {
        "data warehouse", "snowflake", "bigquery", "redshift", "databricks",
        "clickhouse", "duckdb", "synapse",
    },
    # Cloud platforms — intentionally kept narrow.
    # aws/azure/gcp are NOT peers of each other here: a JD requiring "AWS" often means
    # AWS-specific services/tools (CDK, Lambda, EKS), not just any cloud platform.
    # Only the generic "cloud" tag acts as a super-category across providers.
    # DO NOT add aws/azure/gcp as peers — it causes false positives like
    # "azure satisfies aws requirement in Terraform or AWS CDK".
    "cloud_generic": {
        "cloud",
    },
    # JavaScript testing frameworks
    "js_test_frameworks": {
        "unit testing", "jest", "vitest", "mocha", "jasmine", "testing library",
    },
    # Python testing frameworks
    "py_test_frameworks": {
        "unit testing", "pytest", "unittest",
    },
    # Java testing frameworks
    "java_test_frameworks": {
        "unit testing", "junit", "testng",
    },
    # API design / communication protocols
    "api_protocols": {
        "rest api", "graphql", "grpc", "trpc", "openapi", "swagger",
    },
    # Version control
    "vcs": {
        "git", "svn", "mercurial",
    },
    # Code review / quality tools
    "code_quality": {
        "code review", "sonarqube", "eslint", "pylint", "checkstyle",
        "prettier", "black", "ruff",
    },
}

# Build a reverse index: skill → which families it belongs to (O(1) lookup).
_SKILL_TO_FAMILIES: dict[str, set[str]] = {}
for _family_name, _members in SKILL_FAMILIES.items():
    for _member in _members:
        _SKILL_TO_FAMILIES.setdefault(_member, set()).add(_family_name)

# Credit awarded when a resume has a peer family member for a required skill.
# Lower than implied (0.85) since siblings are substitutes, not supersets.
FAMILY_PEER_CREDIT: float = 0.65


def get_family_peers(skill: str) -> set[str]:
    """Return all peer skills that share at least one family with `skill`.

    E.g. get_family_peers("postgresql") → {"sql", "mysql", "sql server", ...}
    (all other members of the sql_databases family, minus the skill itself)
    """
    families = _SKILL_TO_FAMILIES.get(skill, set())
    peers: set[str] = set()
    for fam in families:
        peers.update(SKILL_FAMILIES[fam])
    peers.discard(skill)
    return peers


# ---------------------------------------------------------------------------
# Reverse implication index: requirement → skills that satisfy it.
# Built once at import time from SKILL_IMPLIES for O(1) lookup.
# ---------------------------------------------------------------------------
_REVERSE_IMPLIES: dict[str, set[str]] = {}


def _build_reverse_implies() -> None:
    """Build reverse index: for each implied skill, which source skills imply it?

    This allows O(1) lookup of "which resume skills satisfy requirement X?"
    instead of iterating the full map every time.
    """
    for skill, implied_list in SKILL_IMPLIES.items():
        if skill not in ALL_KNOWN_SKILLS:
            continue
        expanded = set()
        queue = list(implied_list)
        while queue:
            impl = queue.pop()
            if impl in expanded:
                continue
            expanded.add(impl)
            queue.extend(SKILL_IMPLIES.get(impl, []))
        for impl in expanded:
            _REVERSE_IMPLIES.setdefault(impl, set()).add(skill)


_build_reverse_implies()


# ---------------------------------------------------------------------------
# Offline cosine similarity engine (TF-IDF character n-grams on skill names).
# Pre-computed at import time — no LLM / no network needed.
# Used by match_skills() to fuzzy-match "React.js" ≈ "React",
# "API Development" ≈ "REST API", "CI/CD Pipelines" ≈ "CI/CD", etc.
# ---------------------------------------------------------------------------
_SKILL_LIST: list[str] = sorted(ALL_KNOWN_SKILLS)
_SKILL_INDEX: dict[str, int] = {s: i for i, s in enumerate(_SKILL_LIST)}

# Character 2-4 grams capture sub-word similarity across naming variants.
_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
_SKILL_TFIDF_MATRIX = _tfidf.fit_transform(_SKILL_LIST)


def _cosine_sim_skill(name_a: str, name_b: str) -> float:
    """Cosine similarity between two known skill names via pre-computed TF-IDF."""
    idx_a = _SKILL_INDEX.get(name_a)
    idx_b = _SKILL_INDEX.get(name_b)
    if idx_a is None or idx_b is None:
        return 0.0
    return float(sklearn_cosine(
        _SKILL_TFIDF_MATRIX[idx_a], _SKILL_TFIDF_MATRIX[idx_b]
    )[0, 0])


def _cosine_match_unknown(text: str) -> list[tuple[str, float]]:
    """Find the best known-skill matches for an arbitrary text string.

    Used for skill names extracted from resumes/JDs that aren't exact matches
    in ALL_KNOWN_SKILLS (e.g. "React.js", "API Development", "Node").
    Returns up to 3 matches sorted by cosine score descending.
    """
    vec = _tfidf.transform([text.lower().strip()])
    sims = sklearn_cosine(vec, _SKILL_TFIDF_MATRIX).flatten()
    top_indices = np.argsort(sims)[-3:][::-1]
    return [
        (_SKILL_LIST[i], float(sims[i]))
        for i in top_indices
        if sims[i] > 0.3
    ]


# DEPRECATED (2026-04-09): Alias map retained for fallback match_skills() only.
# New skill extraction infers aliases contextually via LLM (extract_skills_llm()).
# Deterministic alias map for common naming variants that character n-grams
# can't reliably distinguish (e.g. abbreviations, brand names, acronyms).
# Maps alias → canonical known skill name.  Both sides must be lowercase.
_SKILL_ALIASES: dict[str, str] = {
    # --- Language aliases ---
    "reactjs": "react", "react.js": "react",
    "vuejs": "vue", "vue.js": "vue",
    "angularjs": "angular", "angular.js": "angular",
    "nodejs": "node.js", "node": "node.js",
    "nextjs": "next.js",
    "nuxtjs": "nuxt.js",
    "expressjs": "express", "express.js": "express",
    "springboot": "spring boot",
    "dotnet": ".net", "dot net": ".net",
    "c sharp": "c#", "csharp": "c#",
    "cpp": "c++", "cplusplus": "c++",
    "golang": "go",
    "obj-c": "objective-c", "objc": "objective-c",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",

    # --- Framework aliases ---
    "nestjs": "nestjs", "nest.js": "nestjs",
    "fastify": "fastify",
    "rails": "rails", "ruby on rails": "rails", "ror": "rails",
    "flask": "flask",
    "django rest framework": "django",
    "asp.net core": "asp.net", "aspnet": "asp.net",
    "spring mvc": "spring",

    # --- Infrastructure / DevOps ---
    "k8s": "kubernetes", "kube": "kubernetes",
    "wcnp": "kubernetes", "walmart cloud native platform": "kubernetes",
    "openshift": "kubernetes",
    "tf": "terraform",
    "aws lambda": "lambda",
    "amazon web services": "aws", "amazon aws": "aws",
    "google cloud platform": "gcp", "google cloud": "gcp",
    "microsoft azure": "azure",
    "docker compose": "docker", "docker swarm": "docker",
    "helm chart": "helm", "helm charts": "helm",
    "github action": "github actions",
    "gitlab ci/cd": "gitlab ci",
    "circle ci": "circleci",
    "argo cd": "argocd", "argo": "argocd",
    "terraform cloud": "terraform",
    "aws cdk": "cdk",
    "aws sam": "sam",
    "cloudwatch": "monitoring",

    # --- Database aliases ---
    "mongo": "mongodb", "mongo db": "mongodb",
    "postgres": "postgresql", "psql": "postgresql",
    "elastic search": "elasticsearch", "elastic": "elasticsearch",
    "dynamo db": "dynamodb", "dynamo": "dynamodb",
    "sql server": "sql server", "mssql": "sql server",
    "maria db": "mariadb",
    "cockroach db": "cockroachdb",
    "influx db": "influxdb",
    "timescale db": "timescaledb",
    "click house": "clickhouse",
    "duck db": "duckdb",
    "neo4j": "neo4j", "neo 4j": "neo4j",

    # --- API aliases ---
    "api development": "rest api", "web api": "rest api",
    "restful": "rest api", "restful api": "rest api",
    "rest apis": "rest api", "api design": "rest api",
    "openapi spec": "openapi", "swagger api": "swagger",

    # --- ML / AI aliases ---
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "machine learning",
    "artificial intelligence": "machine learning",
    "natural language processing": "nlp",
    "cv": "computer vision",
    "sci-kit learn": "scikit-learn", "sklearn": "scikit-learn",
    "sk-learn": "scikit-learn",
    "gen ai": "generative ai", "genai": "generative ai",
    "large language model": "llm", "large language models": "llm",
    "llms": "llm",
    "retrieval augmented generation": "rag",
    "huggingface": "hugging face", "hf": "hugging face",
    "open ai": "openai api", "chatgpt": "openai api",
    "gpt-4": "gpt", "gpt-3": "gpt", "gpt4": "gpt", "gpt3": "gpt",
    "claude": "anthropic api",
    "gemini": "gemini api",
    "neural network": "neural networks", "ann": "neural networks",
    "convolutional neural network": "cnn",
    "recurrent neural network": "rnn",
    "long short term memory": "lstm",
    "generative adversarial network": "gan", "gans": "gan",
    "reinforcement learning": "reinforcement learning", "rl": "reinforcement learning",
    "feature store": "feature engineering",
    "model serving": "model deployment", "model inference": "model deployment",
    "weights and biases": "wandb", "w&b": "wandb",

    # --- Data engineering aliases ---
    "apache spark": "spark", "pyspark": "spark",
    "apache kafka": "kafka",
    "apache airflow": "airflow",
    "apache flink": "flink",
    "apache beam": "beam",
    "apache hive": "hive",
    "apache hadoop": "hadoop",
    "apache nifi": "nifi",
    "extract transform load": "etl",
    "data warehouse": "data warehouse", "dwh": "data warehouse",
    "data lake": "data lake", "datalake": "data lake",
    "data lakehouse": "lakehouse",
    "change data capture": "cdc",

    # --- Data viz aliases ---
    "data viz": "data visualization", "dataviz": "data visualization",
    "bi": "business intelligence",
    "powerbi": "power bi",
    "looker studio": "looker",

    # --- CI/CD / Methodology ---
    "ci cd": "ci/cd", "cicd": "ci/cd",
    "continuous integration": "ci/cd", "continuous deployment": "ci/cd",
    "continuous delivery": "ci/cd",
    "iac": "infrastructure as code", "infra as code": "infrastructure as code",
    "oop": "design patterns", "object oriented": "design patterns",
    "tdd": "test driven development",
    "bdd": "bdd", "behavior driven development": "bdd",
    "ddd": "domain driven design",
    "solid": "solid principles",

    # --- Testing aliases ---
    "qa": "quality assurance",
    "automation": "automation testing",
    "e2e testing": "end-to-end testing", "e2e": "end-to-end testing",
    "perf testing": "performance testing",
    "load test": "load testing",
    "stress testing": "performance testing",
    "api tests": "api testing",
    "unit tests": "unit testing",
    "integration tests": "integration testing",
    "functional testing": "manual testing",
    "acceptance testing": "uat",
    "test automation": "automation testing",

    # --- Security / Auth aliases ---
    "pen testing": "penetration testing", "pentest": "penetration testing",
    "infosec": "cybersecurity", "information security": "cybersecurity",
    "appsec": "application security",
    "netsec": "network security",
    "cloudsec": "cloud security",
    "static analysis": "sast",
    "dynamic analysis": "dast",
    "oauth2": "oauth 2.0", "oauth2.0": "oauth 2.0",
    "json web token": "jwt", "json web tokens": "jwt",
    "bearer token": "jwt", "bearer tokens": "jwt",
    "openid": "openid connect", "oidc": "openid connect",
    "active directory": "azure ad", "ad fs": "azure ad",
    "amazon cognito": "cognito",
    "identity provider": "iam", "idp": "iam",
    "role based access": "iam", "rbac": "iam",
    "attribute based access": "iam", "abac": "iam",

    # --- DevOps / SRE / IaC ---
    "devops engineering": "devops",
    "site reliability": "site reliability", "sre": "site reliability",
    "scrum master": "scrum",
    "platform engineering": "devops",
    "aws cloudformation": "cloudformation",
    "aws cdk": "cdk",
    "azure bicep": "bicep",
    "ansible playbook": "ansible", "ansible playbooks": "ansible",
    "infra as code": "infrastructure as code",

    # --- Mobile aliases ---
    "rn": "react native",
    "swift ui": "swiftui",
    "jetpack": "jetpack compose",
    "kotlin multiplatform mobile": "kotlin multiplatform",
    "kmm": "kotlin multiplatform",

    # --- Tools aliases ---
    "vscode": "vs code", "visual studio code": "vs code",
    "intellij idea": "intellij",
    "neovim": "neovim", "nvim": "neovim",
    "github copilot": "github copilot", "copilot": "github copilot",

    # --- Blockchain aliases ---
    "solidity": "solidity", "sol": "solidity",
    "web 3": "web3", "web 3.0": "web3",
    "smart contract": "smart contracts",

    # --- Misc aliases ---
    "excel": "excel", "ms excel": "excel", "microsoft excel": "excel",
    "gsheets": "google sheets", "google sheet": "google sheets",
    "ppt": "presentation", "powerpoint": "presentation",
}

# Build a reverse map: for each canonical skill, which aliases point to it?
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _alias, _canon in _SKILL_ALIASES.items():
    _ALIAS_TO_CANONICAL[_alias] = _canon


def _resolve_alias(skill_name: str) -> str:
    """Resolve a skill alias to its canonical form if one exists."""
    return _ALIAS_TO_CANONICAL.get(skill_name, skill_name)


def cosine_find_best_match(
    required_skill: str,
    candidate_skills: set[str],
) -> tuple[str | None, float]:
    """Find the best match for a required skill among candidate skills.

    First checks the deterministic alias map (instant, perfect accuracy),
    then falls back to cosine similarity on TF-IDF character n-grams.
    Returns (matched_skill_name, cosine_score) or (None, 0.0).
    """
    # Layer 1: Check if required_skill has an alias that's in candidate_skills
    canonical = _resolve_alias(required_skill)
    if canonical != required_skill and canonical in candidate_skills:
        return canonical, 1.0
    # Check if any candidate is an alias of the required skill
    for cand in candidate_skills:
        if _resolve_alias(cand) == required_skill:
            return cand, 1.0

    # Layer 2: Cosine similarity on TF-IDF vectors
    req_idx = _SKILL_INDEX.get(required_skill)
    if req_idx is None:
        return None, 0.0

    best_name: str | None = None
    best_score = 0.0
    for cand in candidate_skills:
        cand_idx = _SKILL_INDEX.get(cand)
        if cand_idx is None:
            continue
        score = float(sklearn_cosine(
            _SKILL_TFIDF_MATRIX[req_idx], _SKILL_TFIDF_MATRIX[cand_idx]
        )[0, 0])
        if score > best_score:
            best_score = score
            best_name = cand

    return best_name, best_score


def _expand_implied_skills(skills: set[str]) -> set[str]:
    """Expand a skill set with all transitively implied skills.

    e.g. {"next.js"} → {"next.js", "react", "javascript", "html", "css"}
    Uses iterative expansion to resolve chains like:
        next.js → react → javascript + html + css
    """
    expanded = set(skills)
    changed = True
    while changed:
        changed = False
        for skill in list(expanded):
            for implied in SKILL_IMPLIES.get(skill, []):
                if implied not in expanded and implied in ALL_KNOWN_SKILLS:
                    expanded.add(implied)
                    changed = True
    return expanded


def get_skills_that_satisfy_requirement(required_skill: str) -> set[str]:
    """Return resume skills that satisfy a JD requirement via implication.

    E.g. JD requires "Java" → resume having "Spring Boot", "Kotlin", "Scala"
    satisfies it. JD requires "TypeScript" → resume having "React", "Angular"
    satisfies it. Uses pre-built reverse index for O(1) lookup.
    """
    req_lower = required_skill.lower().strip()
    return set(_REVERSE_IMPLIES.get(req_lower, set()))


# ---------------------------------------------------------------------------
# Regex patterns — built once, cached for performance
# ---------------------------------------------------------------------------

# Skills that need special regex handling (not plain \b word boundaries)
_SPECIAL_PATTERNS: dict[str, re.Pattern] = {}
_STANDARD_PATTERNS: dict[str, re.Pattern] = {}


def _has_nonword_chars(s: str) -> bool:
    """Check if skill contains non-word chars (+, #, .) that break \\b."""
    return bool(re.search(r"[+#.]", s))


def _build_patterns() -> None:
    """Pre-compile regex patterns for all skills with correct boundary logic.

    \\b (word boundary) only works between \\w and \\W characters.
    Skills like 'c++', 'c#', '.net', 'node.js' contain non-word chars
    that break \\b, so they need custom boundary patterns.
    """
    # Boundary that works universally: whitespace, punctuation, or string edge
    # (but NOT +, #, . which are part of skill names)
    _LEFT_B = r"(?:^|(?<=[\s,;:(|/]))"
    _RIGHT_B = r"(?=[\s,;:)|/]|$)"

    for skill in ALL_KNOWN_SKILLS:
        if skill.startswith("."):
            # ".net" — must be preceded by whitespace/start
            _SPECIAL_PATTERNS[skill] = re.compile(
                rf"(?:^|(?<=\s)){re.escape(skill)}{_RIGHT_B}"
            )
        elif len(skill) <= 2 and skill.isalpha():
            # Short alpha skills (c, r, go) — strict standalone matching
            _SPECIAL_PATTERNS[skill] = re.compile(
                rf"{_LEFT_B}{re.escape(skill)}{_RIGHT_B}"
            )
        elif _has_nonword_chars(skill):
            # Skills with +, #, . (c++, c#, node.js, ci/cd, etc.)
            # Can't use \b — use explicit boundary lookarounds
            _SPECIAL_PATTERNS[skill] = re.compile(
                rf"(?:^|(?<=[\s,;:(|/])){re.escape(skill)}{_RIGHT_B}"
            )
        else:
            # Normal alpha-numeric skills — standard \b works fine
            _STANDARD_PATTERNS[skill] = re.compile(
                rf"\b{re.escape(skill)}\b"
            )


_build_patterns()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip noise chars (keep +#. for C++, C#, .NET)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s,;:()|-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Extract & match
# ---------------------------------------------------------------------------

def _extract_skills_directly(text: str) -> set[str]:
    """Extract only the skills explicitly mentioned in text (regex + alias resolution).

    Does NOT run implication expansion — returns only what is literally present.
    Used for JD parsing: the JD's explicit requirements should not be inflated by
    what those skills *imply*, otherwise implied sub-skills (e.g. javascript, html, css
    from 'React') become spurious requirements that penalise candidates unfairly.

    Multi-word alias matching takes priority over constituent words:
    "AWS CDK" → canonical "cdk"; the standalone "aws" hit within "aws cdk" is suppressed.
    """
    normalized = _normalize(text)

    _LEFT_B_ALIAS = r"(?:^|(?<=[\s,;:(|/]))"
    _RIGHT_B_ALIAS = r"(?=[\s,;:)|/]|$)"

    # First pass: find multi-word alias spans so their constituent words can be suppressed.
    alias_spans: list[tuple[int, int]] = []
    alias_found: set[str] = set()
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if " " not in alias:
            continue
        if canonical not in ALL_KNOWN_SKILLS:
            continue
        if _has_nonword_chars(alias) or len(alias) <= 2:
            alias_pat = re.compile(
                _LEFT_B_ALIAS + re.escape(alias) + _RIGHT_B_ALIAS, re.IGNORECASE
            )
        else:
            alias_pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        for m in alias_pat.finditer(normalized):
            alias_spans.append((m.start(), m.end()))
            alias_found.add(canonical)

    def _in_alias_span(start: int, end: int) -> bool:
        return any(als <= start and end <= ale for als, ale in alias_spans)

    # Second pass: regex pattern hits — suppress any hit covered by a multi-word alias span.
    directly_found: set[str] = set(alias_found)  # start with alias canonicals

    for skill, pattern in _STANDARD_PATTERNS.items():
        for m in pattern.finditer(normalized):
            if not _in_alias_span(m.start(), m.end()):
                directly_found.add(skill)

    for skill, pattern in _SPECIAL_PATTERNS.items():
        for m in pattern.finditer(normalized):
            if not _in_alias_span(m.start(), m.end()):
                directly_found.add(skill)

    # Single-word alias resolution (not already handled above)
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if " " in alias:
            continue  # already handled in first pass
        if canonical in directly_found:
            continue
        if canonical not in ALL_KNOWN_SKILLS:
            continue
        if _has_nonword_chars(alias) or len(alias) <= 2:
            alias_pattern = re.compile(
                _LEFT_B_ALIAS + re.escape(alias) + _RIGHT_B_ALIAS,
                re.IGNORECASE,
            )
        else:
            alias_pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        if alias_pattern.search(normalized):
            directly_found.add(canonical)

    return directly_found


def extract_skills(text: str) -> set[str]:
    """Extract skills from free text via regex, alias resolution, then implication expansion.

    Use this for RESUME text — the expansion credits the candidate for implied knowledge
    (e.g. 'Next.js' → candidate also knows React, JavaScript, TypeScript).

    For JD text use _extract_skills_directly() to avoid turning implied sub-skills into
    spurious requirements.
    """
    normalized = _normalize(text)
    directly_found: set[str] = set()

    for skill, pattern in _STANDARD_PATTERNS.items():
        if pattern.search(normalized):
            directly_found.add(skill)

    for skill, pattern in _SPECIAL_PATTERNS.items():
        if pattern.search(normalized):
            directly_found.add(skill)

    # Resolve aliases: scan for known alias phrases in the text.
    # Use explicit boundary lookarounds for aliases that start with non-word chars
    # (e.g. ".net") where \b does not work reliably.
    _LEFT_B_ALIAS = r"(?:^|(?<=[\s,;:(|/]))"
    _RIGHT_B_ALIAS = r"(?=[\s,;:)|/]|$)"
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if canonical in directly_found:
            continue
        if canonical not in ALL_KNOWN_SKILLS:
            continue
        if _has_nonword_chars(alias) or len(alias) <= 2:
            alias_pattern = re.compile(
                _LEFT_B_ALIAS + re.escape(alias) + _RIGHT_B_ALIAS,
                re.IGNORECASE,
            )
        else:
            alias_pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        if alias_pattern.search(normalized):
            directly_found.add(canonical)

    # Expand with implied skills
    return _expand_implied_skills(directly_found)


# ---------------------------------------------------------------------------
# Required vs preferred (must-have vs nice-to-have) — regex-only
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _split_jd_required_preferred(jd_text: str) -> tuple[str, str]:
    """Split JD into required and preferred segments by section headers. Returns (required_text, preferred_text)."""
    if not jd_text or not jd_text.strip():
        return ("", "")
    text = jd_text.strip()
    required_parts: list[str] = []
    preferred_parts: list[str] = []
    # Section headers (case-insensitive)
    required_headers = re.compile(
        r"(?:^|\n|\.)\s*(?:required|must\s+have|requirements|qualifications?)\s*[:\-]\s*",
        re.IGNORECASE,
    )
    preferred_headers = re.compile(
        r"(?:^|\n|\.)\s*(?:preferred|nice\s+to\s+have|bonus|pluses?)\s*[:\-]\s*",
        re.IGNORECASE,
    )
    # Split by preferred first, then required, so we can assign chunks
    # Simple approach: find all header positions and assign text between them
    positions: list[tuple[str, int]] = []  # "required" or "preferred", start pos
    for m in required_headers.finditer(text):
        positions.append(("required", m.start()))
    for m in preferred_headers.finditer(text):
        positions.append(("preferred", m.start()))
    positions.sort(key=lambda x: x[1])
    if not positions:
        return (text, "")  # no sections → treat all as required
    # Any text BEFORE the first section header belongs to required (JD preamble/intro)
    preamble_end = positions[0][1]
    if preamble_end > 0:
        required_parts.append(text[:preamble_end].strip())
    for i, (kind, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        chunk = text[start:end].strip()
        if kind == "required":
            required_parts.append(chunk)
        else:
            preferred_parts.append(chunk)
    return (" ".join(required_parts), " ".join(preferred_parts))


# ---------------------------------------------------------------------------
# "or" / "/" alternative detection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _extract_alternative_groups(jd_text: str) -> list[set[str]]:
    """Detect 'X or Y' and 'X/Y' patterns in JD text.

    Strategy: find all known skills in the JD with their text positions,
    then check if any pair of adjacent skills is connected by ' or ' or '/'.
    Groups connected skills as alternatives.

    'Java or Python'       → [{'java', 'python'}]
    'AWS or Azure or GCP'  → [{'aws', 'azure', 'gcp'}]
    'React/Angular/Vue'    → [{'react', 'angular', 'vue'}]
    'ci/cd'                → (skipped, it's a single skill)
    """
    normalized = _normalize(jd_text)

    # Step 1: Find all skill occurrences with positions
    skill_hits: list[tuple[str, int, int]] = []  # (skill_name, start, end)

    for skill, pattern in _STANDARD_PATTERNS.items():
        for m in pattern.finditer(normalized):
            skill_hits.append((skill, m.start(), m.end()))

    for skill, pattern in _SPECIAL_PATTERNS.items():
        for m in pattern.finditer(normalized):
            skill_hits.append((skill, m.start(), m.end()))

    # Step 1b: Also find alias matches and mark their spans so constituent words
    # don't create spurious shorter matches.  E.g. "AWS CDK" is an alias for "cdk";
    # the span of "aws cdk" should suppress the standalone "aws" hit within it.
    alias_spans: list[tuple[int, int, str]] = []  # (start, end, canonical)
    _LEFT_B_ALIAS = r"(?:^|(?<=[\s,;:(|/]))"
    _RIGHT_B_ALIAS = r"(?=[\s,;:)|/]|$)"
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if " " not in alias:
            continue  # single-word aliases already handled by pattern matching above
        if canonical not in ALL_KNOWN_SKILLS:
            continue
        if _has_nonword_chars(alias) or len(alias) <= 2:
            alias_pat = re.compile(
                _LEFT_B_ALIAS + re.escape(alias) + _RIGHT_B_ALIAS, re.IGNORECASE
            )
        else:
            alias_pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        for m in alias_pat.finditer(normalized):
            alias_spans.append((m.start(), m.end(), canonical))
            # Add the canonical skill as a hit at this span position
            skill_hits.append((canonical, m.start(), m.end()))

    # Remove shorter skill hits whose spans are fully contained within an alias span.
    # This prevents "aws" + "cdk" from being treated as independent hits when the
    # text actually says "aws cdk" (which resolves to the single canonical "cdk").
    if alias_spans:
        filtered: list[tuple[str, int, int]] = []
        for skill, s, e in skill_hits:
            covered = any(
                als <= s and e <= ale
                for (als, ale, _canon) in alias_spans
                # Only suppress if the hit is NOT the canonical skill of that alias
                # (we want to keep the canonical hit, not the constituent word hits)
                if _ALIAS_TO_CANONICAL.get(
                    normalized[als:ale].strip().lower(), normalized[als:ale].strip().lower()
                ) != skill
            )
            if not covered:
                filtered.append((skill, s, e))
        skill_hits = filtered

    # Sort by position in text
    skill_hits.sort(key=lambda x: x[1])

    # Step 2: Check text between consecutive skills for ' or ' / '/'
    groups: list[set[str]] = []
    seen: set[str] = set()
    current_group: set[str] = set()

    for i in range(len(skill_hits)):
        skill_a, _, end_a = skill_hits[i]

        if i + 1 < len(skill_hits):
            skill_b, start_b, _ = skill_hits[i + 1]
            between = normalized[end_a:start_b].strip()

            # Check if the text between two skills is " or " or "/"
            is_or = between in ("or", "/")
            # Also match patterns like ", or" or " ,or"
            is_or = is_or or re.fullmatch(r",?\s*or\s*,?", between) is not None

            if is_or and skill_a != skill_b:
                if not current_group:
                    current_group.add(skill_a)
                current_group.add(skill_b)
            else:
                # Chain broken — flush current group if valid.
                # Remove only members already claimed by a previous group, then
                # emit the remaining members as a new group (even if partial).
                if len(current_group) >= 2:
                    new_members = current_group - seen
                    effective_group = new_members if len(new_members) >= 2 else current_group
                    groups.append(effective_group)
                    seen.update(effective_group)
                current_group = set()
        else:
            # Last skill — flush
            if len(current_group) >= 2:
                new_members = current_group - seen
                effective_group = new_members if len(new_members) >= 2 else current_group
                groups.append(effective_group)
                seen.update(effective_group)

    # Handle slash-separated compound terms like "React/Angular/Vue"
    slash_re = re.compile(r"\b([a-z][a-z0-9+#.]*(?:/[a-z][a-z0-9+#.]*)+)\b")
    for match in slash_re.finditer(normalized):
        full_term = match.group(0)
        # Skip if the whole slash-term is a known single skill (e.g. ci/cd)
        if full_term in ALL_KNOWN_SKILLS or full_term in _SPECIAL_PATTERNS:
            continue
        parts = full_term.split("/")
        skills_in_group = {
            p for p in parts
            if p in ALL_KNOWN_SKILLS or p in _SPECIAL_PATTERNS
        }
        if len(skills_in_group) >= 2:
            new_members = skills_in_group - seen
            effective_group = new_members if len(new_members) >= 2 else skills_in_group
            groups.append(effective_group)
            seen.update(effective_group)

    # Deduplicate groups that have identical membership (can arise when both the
    # sequential-scan and the slash-regex detect the same pair, e.g. "JWT/OAuth").
    seen_frozen: set[frozenset[str]] = set()
    unique_groups: list[set[str]] = []
    for g in groups:
        fg = frozenset(g)
        if fg not in seen_frozen:
            seen_frozen.add(fg)
            unique_groups.append(g)
    return unique_groups


def match_skills(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
) -> dict[str, Any]:
    """Compare resume skills against JD + role profile skills.

    Handles 'X or Y' patterns: when the JD says 'Java or Python',
    matching either one counts as a full match for that requirement.

    When resume_text or jd_text is empty/whitespace, returns score 0 with
    empty matched/missing lists and total_required=0 (displayed as 0/0 or 0/1).
    """
    resume_text = (resume_text or "").strip()
    jd_text = (jd_text or "").strip()

    # Resume: expand via implication so candidate gets credit for implied knowledge.
    # e.g. 'Next.js' → candidate also knows React, JavaScript, TypeScript, HTML, CSS.
    resume_skills = extract_skills(resume_text)

    req_text, pref_text = _split_jd_required_preferred(jd_text)
    use_required_preferred = bool(pref_text.strip())

    # JD: extract ONLY explicitly mentioned skills — do NOT expand via implication.
    # Expanding would turn 'Kubernetes' into {kubernetes, docker, linux} requirements,
    # penalising candidates for docker/linux even though the JD never asked for them.
    # The implication direction is reversed here: if resume has 'eks' that implies kubernetes,
    # the reverse-implication index (get_skills_that_satisfy_requirement) handles that.
    if use_required_preferred:
        jd_skills_required = _extract_skills_directly(req_text)
        jd_skills_preferred = _extract_skills_directly(pref_text)
    else:
        jd_skills_required = _extract_skills_directly(jd_text)
        jd_skills_preferred = set()

    # Detect alternative groups from required segment (or full JD if no split)
    jd_for_alt = req_text if use_required_preferred else jd_text
    alt_groups = _extract_alternative_groups(jd_for_alt)
    alt_skill_set = set()
    for group in alt_groups:
        alt_skill_set.update(group)

    # Build required skill set from JD + role profile
    required_flat = set(jd_skills_required) - alt_skill_set
    role_key = (role or "").lower().strip()
    if role_key in ROLE_PROFILES:
        for s in ROLE_PROFILES[role_key]["skills"]:
            normalized = s.lower().strip()
            if normalized not in alt_skill_set:
                required_flat.add(normalized)

    from app.config import SKILL_COSINE_FULL_MATCH, SKILL_COSINE_PARTIAL_MATCH

    # --- Score flat (non-alternative) required skills ---
    # Layer 1: Direct exact matches
    matched_flat_direct = resume_skills & required_flat

    # Layer 2: Implied matches via transitive implication graph.
    # Example: JD requires "java" → resume has "spring boot" → spring boot implies java → match.
    matched_flat_implied: set[str] = set()
    for req_skill in (required_flat - matched_flat_direct):
        satisfiers = get_skills_that_satisfy_requirement(req_skill)
        if resume_skills & satisfiers:
            matched_flat_implied.add(req_skill)

    # Layer 3: Family peer matching — same-purpose technology family.
    # Example: JD requires "postgresql" → resume has "mysql" (same SQL DB family) → partial credit.
    # Example: JD requires "terraform" → resume has "cdk" (same IaC family) → partial credit.
    # Example: JD requires "oauth 2.0" → resume has "jwt" (same auth family) → partial credit.
    # This also handles upward generalization: resume has "azure sql" → implies "sql" →
    # "sql" is in the same sql_databases family as "postgresql" → peer match.
    matched_flat_family: dict[str, float] = {}  # req_skill → family peer credit
    still_after_implied = required_flat - matched_flat_direct - matched_flat_implied
    for req_skill in still_after_implied:
        peers = get_family_peers(req_skill)
        # Check if resume contains any peer directly or via implication expansion
        # (e.g. resume has "azure sql" which implies "sql", and "sql" is a peer of "postgresql")
        peer_hit = resume_skills & peers
        if not peer_hit:
            # Also check expanded resume skills — catches cases where the resume skill
            # implies something that is a family peer (e.g. bigquery implies sql)
            for rs in resume_skills:
                rs_implied = _expand_implied_skills({rs})
                if rs_implied & peers:
                    peer_hit = {rs}
                    break
        if peer_hit:
            matched_flat_family[req_skill] = FAMILY_PEER_CREDIT

    # Layer 4: Cosine similarity fuzzy matching for remaining misses.
    # Catches name variants not covered by alias map (e.g. "React.js" vs "React").
    matched_flat_cosine_full: set[str] = set()
    matched_flat_cosine_partial: dict[str, float] = {}  # skill → cosine score
    still_missing = still_after_implied - set(matched_flat_family)
    if still_missing and resume_skills:
        for req_skill in still_missing:
            best_name, best_cos = cosine_find_best_match(req_skill, resume_skills)
            if best_name and best_cos >= SKILL_COSINE_FULL_MATCH:
                matched_flat_cosine_full.add(req_skill)
            elif best_name and best_cos >= SKILL_COSINE_PARTIAL_MATCH:
                matched_flat_cosine_partial[req_skill] = best_cos

    matched_flat = matched_flat_direct | matched_flat_implied | matched_flat_cosine_full | set(matched_flat_family)
    missing_flat = required_flat - matched_flat - set(matched_flat_cosine_partial)

    # --- Score alternative groups ---
    matched_alt_skills = set()
    missing_alt_labels: list[str] = []
    alt_matched_count = 0
    for group in alt_groups:
        candidate_has = resume_skills & group
        if not candidate_has:
            for g_skill in group:
                satisfiers = get_skills_that_satisfy_requirement(g_skill)
                if resume_skills & satisfiers:
                    candidate_has = resume_skills & satisfiers
                    break
        if not candidate_has:
            # Also try family peer matching for alt groups
            for g_skill in group:
                peers = get_family_peers(g_skill)
                peer_hit = resume_skills & peers
                if not peer_hit:
                    for rs in resume_skills:
                        if _expand_implied_skills({rs}) & peers:
                            peer_hit = {rs}
                            break
                if peer_hit:
                    candidate_has = peer_hit
                    break
        if not candidate_has:
            for g_skill in group:
                best_name, best_cos = cosine_find_best_match(g_skill, resume_skills)
                if best_name and best_cos >= SKILL_COSINE_FULL_MATCH:
                    candidate_has = {best_name}
                    break
        if candidate_has:
            alt_matched_count += 1
            matched_alt_skills.update(candidate_has)
        else:
            missing_alt_labels.append(" or ".join(sorted(group)))

    # --- Combine required score ---
    total_requirements = len(required_flat) + len(alt_groups)
    total_matched_required = len(matched_flat) + alt_matched_count

    # Credit tiers (descending):
    #   direct=1.0, implied=0.85, family-peer=FAMILY_PEER_CREDIT(0.65),
    #   cosine-full=0.80, cosine-partial=scaled 0.7×cosine_score
    # Note: alt_matched_count uses full 1.0 credit per alternative group matched.
    implied_count = len(matched_flat_implied)
    family_peer_credit = sum(matched_flat_family.values())
    cosine_full_count = len(matched_flat_cosine_full)
    cosine_partial_credit = sum(
        0.7 * score for score in matched_flat_cosine_partial.values()
    )
    flat_direct_count = len(matched_flat_direct)
    effective_matched = (
        flat_direct_count
        + (implied_count * 0.85)
        + family_peer_credit
        + (cosine_full_count * 0.80)
        + cosine_partial_credit
        + (alt_matched_count * 0.85)  # alt group matches get 0.85 (OR logic = slightly penalized)
    )
    required_ratio = min(
        (effective_matched / total_requirements) if total_requirements else 1.0,
        1.0,  # clamp to 1.0 — never exceed 100% for required skills
    )

    # --- Preferred (nice-to-have) score when JD has required/preferred sections ---
    if use_required_preferred and jd_skills_preferred:
        preferred_flat = jd_skills_preferred - alt_skill_set
        matched_preferred: float = float(len(resume_skills & preferred_flat))
        for pref_skill in (preferred_flat - resume_skills):
            if resume_skills & get_skills_that_satisfy_requirement(pref_skill):
                matched_preferred += 0.85
                continue
            # Family peer matching for preferred skills too
            pref_peers = get_family_peers(pref_skill)
            pref_peer_hit = resume_skills & pref_peers
            if not pref_peer_hit:
                for rs in resume_skills:
                    if _expand_implied_skills({rs}) & pref_peers:
                        pref_peer_hit = {rs}
                        break
            if pref_peer_hit:
                matched_preferred += FAMILY_PEER_CREDIT
                continue
            _, pref_cos = cosine_find_best_match(pref_skill, resume_skills)
            if pref_cos >= SKILL_COSINE_FULL_MATCH:
                matched_preferred += 0.80
            elif pref_cos >= SKILL_COSINE_PARTIAL_MATCH:
                matched_preferred += 0.5
        total_preferred = max(len(preferred_flat), 1)
        preferred_ratio = min(matched_preferred / total_preferred, 1.0)
        score = round((0.7 * required_ratio + 0.3 * preferred_ratio) * 100, 2)
    else:
        total = max(total_requirements, 1)
        score = round(min((effective_matched / total) * 100, 100.0), 2)

    all_matched = sorted(matched_flat | matched_alt_skills)
    all_missing = sorted(missing_flat) + sorted(missing_alt_labels)
    extra = sorted(resume_skills - required_flat - alt_skill_set)

    cosine_matches_detail = [
        {"required": sk, "cosine": round(sc, 3)}
        for sk, sc in matched_flat_cosine_partial.items()
    ]
    family_matches_detail = [
        {"required": sk, "credit": round(cr, 3), "match_type": "family_peer"}
        for sk, cr in matched_flat_family.items()
    ]

    return {
        "matched": all_matched,
        "missing": all_missing,
        "extra": extra,
        "score": score,
        "total_required": total_requirements,
        "matched_count": total_matched_required,
        "cosine_partial_matches": cosine_matches_detail,
        "family_peer_matches": family_matches_detail,
    }


def get_available_roles() -> list[str]:
    """Return list of supported role names."""
    return sorted(ROLE_PROFILES.keys())


def get_role_weights(role: str | None = None) -> dict[str, float]:
    """Return scoring weights. When USE_ROLE_WEIGHTS is true and role is set, build a
    weight dict exclusively from role_profiles values (normalized to sum 1.0); otherwise
    use UNIFIED_WEIGHTS as-is.

    Role profile weights can be stored as fractions (0.40) or percentages (40). Both are
    normalized to fractions before use. Dimensions missing from the role profile are filled
    from UNIFIED_WEIGHTS at their relative proportion, then the whole dict is re-normalized.
    """
    from app.config import UNIFIED_WEIGHTS, USE_ROLE_WEIGHTS
    base = UNIFIED_WEIGHTS.copy()
    if USE_ROLE_WEIGHTS and role and (role_key := (role or "").strip().lower()) and role_key in ROLE_PROFILES:
        rw = ROLE_PROFILES[role_key].get("weights") or {}
        if rw:
            # Normalize role profile values: if any value > 1 treat all as percentages
            raw_values = {k: float(v) for k, v in rw.items() if k in base}
            if raw_values and max(raw_values.values()) > 1.0:
                raw_values = {k: v / 100.0 for k, v in raw_values.items()}
            # Apply role profile values; keep UNIFIED_WEIGHTS for missing keys
            for k, v in raw_values.items():
                base[k] = v
            # Re-normalize the combined dict so weights always sum to exactly 1.0
            total = sum(base.values())
            if total > 0:
                base = {k: v / total for k, v in base.items()}
    return base
