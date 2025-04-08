"""update_model_table

Revision ID: dcf3bbcfd8f5
Revises: 56ca0f2dff70
Create Date: 2025-03-27 20:29:23.541997

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'dcf3bbcfd8f5'
down_revision = '56ca0f2dff70'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop existing models_new table if it exists
    op.execute("DROP TABLE IF EXISTS models_new")
    
    # Create new table with desired schema
    op.create_table(
        'models_new',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(500)),
        sa.Column('model_type', sa.String(50)),
        sa.Column('model_framework', sa.String(50), nullable=False),
        sa.Column('version', sa.String(20)),
        sa.Column('file_path', sa.String(500)),
        sa.Column('project_id', sa.Integer, sa.ForeignKey('projects.id')),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id')),
        sa.Column('model_metadata', sa.JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    # Copy data from old table to new table
    op.execute("""
        INSERT INTO models_new 
        SELECT id, name, description, model_type, 
            CASE 
                WHEN model_type = 'classification' OR model_type = 'regression' THEN 'sklearn'
                WHEN model_type = 'nlp' THEN 'tensorflow'
                ELSE 'onnx'
            END as model_framework,
            version, file_path, project_id, user_id, model_metadata,
            created_at, updated_at
        FROM models
    """)

    # Drop old table
    op.drop_table('models')

    # Rename new table to original name
    op.rename_table('models_new', 'models')


def downgrade() -> None:
    # Drop existing models_old table if it exists
    op.execute("DROP TABLE IF EXISTS models_old")
    
    # Create old table structure
    op.create_table(
        'models_old',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(500)),
        sa.Column('model_type', sa.String(50)),
        sa.Column('version', sa.String(20)),
        sa.Column('file_path', sa.String(500)),
        sa.Column('project_id', sa.Integer, sa.ForeignKey('projects.id')),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id')),
        sa.Column('model_metadata', sa.JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    # Copy data back
    op.execute("""
        INSERT INTO models_old 
        SELECT id, name, description, model_type, version, file_path, 
               project_id, user_id, model_metadata, created_at, updated_at
        FROM models
    """)

    # Drop new table
    op.drop_table('models')

    # Rename old table to original name
    op.rename_table('models_old', 'models') 