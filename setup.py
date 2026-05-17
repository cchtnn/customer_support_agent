import os

# Base project structure
project_structure = {
    "customer_support_agent": {
        "backend": {
            "agents": [
                "intent_classifier.py",
                "sentiment_priority.py",
                "rag_retrieval.py",
                "response_generator.py",
                "escalation.py",
                "analytics.py",
                "qa_compliance.py",
            ],
            "graph": [
                "orchestration.py",
            ],
            "rag": [
                "vector_db.py",
            ],
            "api": [
                "main.py",
                "routes.py",
                "schemas.py",
            ],
            "utils": [
                "config.py",
            ],
        },

        "frontend": [
            "app.py",
        ],

        "data": {
            "chroma_db": []
        },

        # Root-level files inside customer_support_agent/
        "__files__": [
            "docker-compose.yml",
            "requirements.txt",
            ".env",
            "README.md",
        ],
    }
}


def create_structure(base_path, structure):
    """
    Recursively create folders and files.
    """

    for name, content in structure.items():

        # Handle root-level files
        if name == "__files__":
            for file_name in content:
                file_path = os.path.join(base_path, file_name)

                with open(file_path, "w") as f:
                    f.write(f"# {file_name}\n")

            continue

        current_path = os.path.join(base_path, name)

        # Create directory
        os.makedirs(current_path, exist_ok=True)

        if isinstance(content, dict):
            # Recursive call
            create_structure(current_path, content)

        elif isinstance(content, list):

            for item in content:

                item_path = os.path.join(current_path, item)

                # Create file
                if "." in item:
                    with open(item_path, "w") as f:
                        f.write(f"# {item}\n")

                # Create folder
                else:
                    os.makedirs(item_path, exist_ok=True)


# Create project structure
create_structure(".", project_structure)

print("✅ customer_support_agent project structure created successfully!")