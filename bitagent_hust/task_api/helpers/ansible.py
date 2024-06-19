from git import Repo
import os
import shutil
import atexit
import yaml
from yaml import CLoader as Loader

# Define a custom loader that ignores unknown tags
class CustomLoader(Loader):
    def ignore_unknown_tags(self, node):
        return None

# Register the custom tag handler to the loader for all unknown tags
CustomLoader.add_constructor(None, CustomLoader.ignore_unknown_tags)

# Function to load and merge YAMLs from a file
def yaml_loader(file_path):
    with open(file_path, 'r') as file:
        # Load all documents
        documents = list(yaml.load_all(file, Loader=CustomLoader))
        # Initialize a base dictionary to merge all documents
        merged_document = {}
        for doc in documents:
            if isinstance(doc, dict):
                # Merge dictionaries
                merged_document.update(doc)
            else:
                # Merge lists
                merged_document = doc if isinstance(doc, list) else [doc]
        return merged_document

class AnsibleRepoAnalyzer:
    def __init__(self, repo_name, access_token, local_base_path='repo'):
        self.repo_url = f'https://github.com/{repo_name}.git'
        self.local_path = f'{local_base_path}/{repo_name}'
        self.access_token = access_token
        self.repo = None
        atexit.register(self.cleanup)  # Register cleanup method to run on program exit
    
    def __enter__(self):
        """Constructor to handle the initialization when the instance is created."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destructor to handle the cleanup when the instance is deleted."""
        self.cleanup()


    def clone_repo(self):
        """Clones the repository to a local directory. Only saves .yml and .yaml files"""
        if os.path.exists(self.local_path):
            return
        if os.path.isdir(os.path.join(self.local_path, '.git')):
            self.repo = Repo(self.local_path)
        else:
            self.repo = Repo.init(self.local_path)

        # Configure sparse-checkout
        with self.repo.config_writer() as git_config:
            git_config.set_value('core', 'sparseCheckout', 'true')
            git_config.set_value('http', 'postBuffer', '524288000')

        # Create or use existing remote
        if 'origin' in self.repo.remotes:
            remote = self.repo.remotes.origin
            remote.set_url(self.repo_url)
        else:
            remote = self.repo.create_remote('origin', self.repo_url)
        with open(f'{self.local_path}/.git/info/sparse-checkout', 'w') as sc_file:
            sc_file.write('*.yml\n')
            sc_file.write('*.yaml\n')
        
        remote.fetch(depth=1, jobs=10)
        try:
            default_branch = self.repo.git.symbolic_ref('refs/remotes/origin/HEAD').split('/')[-1]
        except Exception as e:
            # If fetching the default branch fails, check for common branch names
            common_branches = ['main', 'master', 'develop', 'dev']
            for branch in common_branches:
                if f'origin/{branch}' in self.repo.refs:
                    default_branch = branch
                    break
            else:
                # If no common branch is found, raise an error or set a default branch
                raise ValueError("Default branch could not be determined. Please specify manually.")

        self.repo.git.checkout(default_branch, env={'GIT_ASKPASS': 'echo', 'GIT_PASSWORD': self.access_token})
            


    def get_files(self, path=''):
        """Recursively get all files in the local repository directory."""
        for root, dirs, files in os.walk(os.path.join(self.local_path, path)):
            for file in files:
                yield os.path.join(root, file)

    def is_ansible_file(self, file_path):
        """Check if a YAML file is an Ansible file by analyzing its contents."""
        try:
            content = yaml_loader(file_path)
            if isinstance(content, list):  # Ansible playbooks are typically lists
                return any('hosts' in item for item in content)
            elif isinstance(content, dict):
                return any(key in content for key in ['tasks', 'handlers', 'roles'])
        except yaml.YAMLError:
            return False
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False
        return False

    def filter_ansible_files(self, files):
        """Filter files to find Ansible playbooks and roles based on their content."""
        return [f for f in files if f.endswith(('.yml', '.yaml')) and self.is_ansible_file(f)]
    
    def extract_tasks(self, content):
        """Extracts tasks from the YAML content."""
        if isinstance(content, list):  # Typical structure of a playbook or a direct list of tasks
            for item in content:
                if isinstance(item, dict) and 'tasks' in item:
                    return item['tasks']
                elif isinstance(item, dict):  # Direct task entries
                    return content
        elif isinstance(content, dict) and 'tasks' in content:  # Direct dictionary with tasks key
            return content['tasks']
        return []  # Return an empty list if no tasks found

    def get_tasks(self, file_path):
        """Extract and return tasks from a YAML file that might contain Ansible tasks."""
        try:
            content = yaml_loader(file_path)
            return self.extract_tasks(content)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in file {file_path}: {e}")
            return []
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    def get_ansible_tasks(self):
        """Get tasks from all YAML files in the local repository that potentially contain Ansible tasks."""
        self.clone_repo()  # Ensure the repository is cloned
        all_files = list(self.get_files())
        tasks = {}
        for file in all_files:
            if file.endswith(('.yml', '.yaml')):
                tasks_content = self.get_tasks(file)
                if tasks_content:
                    relative_path = os.path.relpath(file, self.local_path)
                    tasks[relative_path] = tasks_content
        return tasks
        
    def cleanup(self):
        """Cleanup the local clone of the repository."""
        if os.path.exists(self.local_path):
            shutil.rmtree(self.local_path)
            print("Cleaned up local repository.")
