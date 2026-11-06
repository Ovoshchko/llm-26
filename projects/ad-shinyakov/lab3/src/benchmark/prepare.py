import json
import subprocess

from .cluster import LAB_ROOT, LocalCluster


class EvaluationSetup:
    def run(self):
        LocalCluster()
        endpoint = subprocess.check_output(['docker', 'context', 'inspect', '--format',
                                            '{{.Endpoints.docker.Host}}'], text=True).strip()
        subprocess.run(['docker', '--config', str(LAB_ROOT / 'data/docker-build'), '--host', endpoint,
                        'build', '-t', 'agent-lab/evaluation:dev', str(LAB_ROOT / 'evaluation/fixture_app')], check=True)
        subprocess.run([str(LAB_ROOT / 'data/bin/k3d'), 'image', 'import', 'agent-lab/evaluation:dev',
                        '-c', 'agent-lab'], check=True)
        image_id = subprocess.check_output(['docker', 'image', 'inspect', 'agent-lab/evaluation:dev',
                                            '--format', '{{.Id}}'], text=True).strip()
        destination = LAB_ROOT / 'data/evaluation/fixture-image.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({'image': 'agent-lab/evaluation:dev', 'image_id': image_id}) + '\n')
        print(destination)


if __name__ == '__main__':
    EvaluationSetup().run()
