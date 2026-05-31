"""Initial schema — all HarchOS v0.7.0 tables.

Revision ID: 001_initial
Revises: None
Create Date: 2026-03-05
"""

from alembic import op
import sqlalchemy as sa

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True, default=True),
        sa.Column("role", sa.String(20), nullable=False, server_default="user"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # --- projects ---
    op.create_table(
        "projects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("tier", sa.String(20), nullable=False, server_default="free"),
        sa.Column("is_active", sa.Boolean(), nullable=True, default=True),
        sa.Column("usage_limits", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"])

    # --- hubs ---
    op.create_table(
        "hubs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("region", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="creating"),
        sa.Column("tier", sa.String(50), nullable=False, server_default="standard"),
        sa.Column("total_gpus", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("available_gpus", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cpu_cores", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("available_cpu_cores", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_memory_gb", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("available_memory_gb", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_storage_gb", sa.Float(), nullable=False, server_default="50000.0"),
        sa.Column("available_storage_gb", sa.Float(), nullable=False, server_default="35000.0"),
        sa.Column("latitude", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("longitude", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("city", sa.String(255), nullable=False, server_default=""),
        sa.Column("country", sa.String(255), nullable=False, server_default=""),
        sa.Column("renewable_percentage", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("grid_carbon_intensity", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("pue", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("sovereignty_level", sa.String(50), nullable=False, server_default="standard"),
        sa.Column("data_residency_policy", sa.String(255), nullable=False, server_default="local_only"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # --- api_keys ---
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("key_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("key_prefix", sa.String(16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("tier", sa.String(20), nullable=False, server_default="free"),
        sa.Column("scopes", sa.Text(), nullable=True),
        sa.Column("allowed_models", sa.Text(), nullable=True),
        sa.Column("allowed_regions", sa.Text(), nullable=True),
        sa.Column("max_tokens_per_day", sa.Integer(), nullable=True),
        sa.Column("tokens_used_today", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("spending_limit_monthly_usd", sa.Float(), nullable=True),
        sa.Column("spent_this_month_usd", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
    op.create_index("ix_api_keys_project_id", "api_keys", ["project_id"])

    # --- models ---
    op.create_table(
        "models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("framework", sa.String(50), nullable=False, server_default="pytorch"),
        sa.Column("task", sa.String(255), nullable=False, server_default=""),
        sa.Column("status", sa.String(50), nullable=False, server_default="draft"),
        sa.Column("capabilities_json", sa.Text(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column("hub_id", sa.String(36), sa.ForeignKey("hubs.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # --- workloads ---
    op.create_table(
        "workloads",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("gpu_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gpu_type", sa.String(100), nullable=False, server_default=""),
        sa.Column("cpu_cores", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("memory_gb", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("storage_gb", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("hub_id", sa.String(36), sa.ForeignKey("hubs.id"), nullable=True),
        sa.Column("priority", sa.String(20), nullable=False, server_default="normal"),
        sa.Column("sovereignty_level", sa.String(50), nullable=False, server_default="standard"),
        sa.Column("data_residency_policy", sa.String(255), nullable=False, server_default=""),
        sa.Column("carbon_aware", sa.Boolean(), nullable=True, default=False),
        sa.Column("carbon_intensity_threshold", sa.Float(), nullable=True),
        sa.Column("carbon_budget_grams", sa.Float(), nullable=True),
        sa.Column("carbon_actual_grams", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_workloads_user_id", "workloads", ["user_id"])

    # --- energy_reports ---
    op.create_table(
        "energy_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("resource_id", sa.String(36), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False, server_default="hub"),
        sa.Column("total_consumption_kwh", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("renewable_percentage", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("carbon_emissions_kg", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("pue", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("efficiency_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_energy_reports_resource_id", "energy_reports", ["resource_id"])

    # --- energy_consumption ---
    op.create_table(
        "energy_consumption",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("resource_id", sa.String(36), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False, server_default="hub"),
        sa.Column("consumption_kwh", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("carbon_emissions_kg", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("renewable_percentage", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_energy_consumption_resource_id", "energy_consumption", ["resource_id"])

    # --- carbon_intensity_records ---
    op.create_table(
        "carbon_intensity_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("zone", sa.String(50), nullable=False),
        sa.Column("carbon_intensity_gco2_kwh", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("renewable_percentage", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("fossil_percentage", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("fuel_mix_json", sa.Text(), nullable=True),
        sa.Column("source", sa.String(100), nullable=False, server_default="electricity_maps"),
        sa.Column("is_forecast", sa.Boolean(), nullable=True, default=False),
        sa.Column("datetime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_carbon_intensity_records_zone", "carbon_intensity_records", ["zone"])

    # --- carbon_optimization_logs ---
    op.create_table(
        "carbon_optimization_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("workload_id", sa.String(36), nullable=True),
        sa.Column("workload_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("selected_hub_id", sa.String(36), nullable=True),
        sa.Column("selected_hub_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("carbon_intensity_at_schedule_gco2_kwh", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("carbon_saved_kg", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("baseline_carbon_kg", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("actual_carbon_kg", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("deferred_hours", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_carbon_optimization_logs_workload_id", "carbon_optimization_logs", ["workload_id"])

    # --- pricing ---
    op.create_table(
        "pricing",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("gpu_type", sa.String(50), nullable=False),
        sa.Column("price_per_gpu_hour", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("price_per_cpu_core_hour", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("price_per_gb_storage_month", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("price_per_gb_memory_hour", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("currency", sa.String(10), nullable=False, server_default="USD"),
        sa.Column("region", sa.String(255), nullable=False, server_default=""),
        sa.Column("tier", sa.String(50), nullable=False, server_default="standard"),
        sa.Column("is_default", sa.Boolean(), nullable=True, default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_pricing_gpu_type", "pricing", ["gpu_type"])

    # --- billing_records ---
    op.create_table(
        "billing_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("workload_id", sa.String(36), sa.ForeignKey("workloads.id"), nullable=True),
        sa.Column("hub_id", sa.String(36), sa.ForeignKey("hubs.id"), nullable=True),
        sa.Column("gpu_hours", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("cpu_core_hours", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("memory_gb_hours", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("storage_gb_months", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("currency", sa.String(10), nullable=False, server_default="USD"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_billing_records_user_id", "billing_records", ["user_id"])
    op.create_index("ix_billing_records_workload_id", "billing_records", ["workload_id"])
    op.create_index("ix_billing_records_hub_id", "billing_records", ["hub_id"])

    # --- batch_jobs ---
    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("total_items", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_items", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_items", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_data", sa.Text(), nullable=False),
        sa.Column("results", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("aggregate_carbon_footprint", sa.Text(), nullable=True),
        sa.Column("carbon_aware", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_batch_jobs_user_id", "batch_jobs", ["user_id"])

    # --- fine_tuning_jobs ---
    op.create_table(
        "fine_tuning_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("method", sa.String(20), nullable=False, server_default="lora"),
        sa.Column("training_file_id", sa.String(64), nullable=False),
        sa.Column("validation_file_id", sa.String(64), nullable=True),
        sa.Column("hyperparameters", sa.Text(), nullable=False),
        sa.Column("carbon_tracking", sa.Text(), nullable=False),
        sa.Column("training_metrics", sa.Text(), nullable=False),
        sa.Column("cost_estimate", sa.Text(), nullable=True),
        sa.Column("fine_tuned_model", sa.String(200), nullable=True),
        sa.Column("suffix", sa.String(64), nullable=True),
        sa.Column("trained_tokens", sa.Integer(), nullable=True),
        sa.Column("epoch", sa.Integer(), nullable=True),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("webhook_url", sa.String(2048), nullable=True),
        sa.Column("webhook_secret", sa.String(128), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_fine_tuning_jobs_user_id", "fine_tuning_jobs", ["user_id"])

    # --- fine_tuning_files ---
    op.create_table(
        "fine_tuning_files",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("purpose", sa.String(50), nullable=False, server_default="fine-tune"),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="processed"),
        sa.Column("status_details", sa.Text(), nullable=True),
        sa.Column("line_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sha256", sa.String(64), nullable=False, server_default=""),
        sa.Column("file_content", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_fine_tuning_files_user_id", "fine_tuning_files", ["user_id"])

    # --- fine_tuned_models ---
    op.create_table(
        "fine_tuned_models",
        sa.Column("id", sa.String(200), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("base_model", sa.String(100), nullable=False),
        sa.Column("fine_tuning_job_id", sa.String(64), nullable=False),
        sa.Column("method", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="ready"),
        sa.Column("carbon_grams", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("suffix", sa.String(64), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_fine_tuned_models_user_id", "fine_tuned_models", ["user_id"])


def downgrade() -> None:
    op.drop_table("fine_tuned_models")
    op.drop_table("fine_tuning_files")
    op.drop_table("fine_tuning_jobs")
    op.drop_table("batch_jobs")
    op.drop_table("billing_records")
    op.drop_table("pricing")
    op.drop_table("carbon_optimization_logs")
    op.drop_table("carbon_intensity_records")
    op.drop_table("energy_consumption")
    op.drop_table("energy_reports")
    op.drop_table("workloads")
    op.drop_table("models")
    op.drop_table("api_keys")
    op.drop_table("projects")
    op.drop_table("hubs")
    op.drop_table("users")
