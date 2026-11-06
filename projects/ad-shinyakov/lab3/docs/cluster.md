# Локальный Kubernetes-стенд

Стенд запускается через k3d в Docker Desktop. Кластер `agent-lab` состоит из одной
ноды k3s с лимитом памяти 2 ГБ. Traefik и ServiceLB отключены; API доступен только
на `127.0.0.1:6550`. Приложения доступны через kubectl port-forward.

Проверенная конфигурация: Apple Silicon, Docker Desktop 28.3.3, k3d 5.9.0,
k3s 1.33.6 и kubectl 1.32.2. Версия k3s зафиксирована в infra/k3d.yaml и выбрана
с учётом установленного клиента kubectl.

## Файлы

| Файл | Назначение |
|---|---|
| [k3d.yaml](../infra/k3d.yaml) | Конфигурация отдельного кластера |
| [workload.yaml](../infra/workload.yaml) | Namespace, ConfigMap, Deployment api и Service |
| [rbac.yaml](../infra/rbac.yaml) | Учётная запись agent и права в agent-lab |
| [Dockerfile](../infra/app/Dockerfile) | Локальный образ приложения |
| [server.py](../infra/app/server.py) | HTTP-приложение с /health и /ready |
| [cluster.sh](../scripts/cluster.sh) | Создание, запуск, развёртывание и остановка |

## Запуск

Требуются работающий Docker Desktop, kubectl и k3d. Скрипт сначала использует
`data/bin/k3d`, если он есть, затем ищет k3d в PATH. Для установки на macOS:

```bash
brew install k3d
```

В подготовленном окружении k3d уже загружен в data/bin из официального релиза 5.9.0,
контрольная сумма SHA-256 проверена. Этот бинарник не входит в Git.

Из каталога lab3:

```bash
bash scripts/cluster.sh up
bash scripts/cluster.sh deploy
bash scripts/cluster.sh status
```

up создаёт отсутствующий кластер либо запускает существующий. deploy собирает
образ `agent-lab/app:dev`, импортирует его в containerd ноды, применяет манифесты
и ждёт готовности Deployment. При imagePullPolicy=Never Pod используют локально
импортированный образ; первый docker build загружает базовый python:3.12-alpine.

Повторный deploy применяет исходную конфигурацию, включая две реплики и версию v1.
Это подготовка базового стенда; во время диалога его вызывать не следует.
Тег dev предназначен для разработки. Скрипт записывает ID собранного образа
в аннотацию шаблона Pod: изменение образа запускает rollout автоматически.
Для публичного базового образа используется отдельный анонимный Docker-конфиг
в data/docker-build, чтобы не обращаться к зависающему credential helper Desktop.

## Доступ

Скрипт хранит отдельный административный kubeconfig в `data/cluster/kubeconfig.yaml`
с правами 600. Папка data исключена из Git. Основной kubeconfig и текущий контекст
не изменяются. Для ручных команд в текущем терминале:

```bash
export KUBECONFIG="$PWD/data/cluster/kubeconfig.yaml"
kubectl -n agent-lab get pods
kubectl -n agent-lab logs deployment/api --tail=10
kubectl -n agent-lab port-forward service/api 18080:80
```

В другом терминале:

```bash
curl http://127.0.0.1:18080/ready
```

Ожидается JSON с `status=ok` и `config_version=v1`. На старте каждый Pod записывает
config_version в stdout. Значение берётся из ConfigMap через envFrom и меняется
для работающего приложения после пересоздания Pod.

## Учётная запись агента

ServiceAccount `agent` имеет Role только в agent-lab: чтение Deployment, ReplicaSet,
Pod, логов, событий, ConfigMap, Service, EndpointSlice и HPA; patch Deployment и
изменение scale. Доступ к Secrets, нодам и другим namespace не выдаётся.
Ограничение числа реплик 1–5 проверяется схемой scale_deployment; restart_deployment
изменяет аннотацию шаблона подов через resources_create_or_update в MCP.
RBAC не ограничивает значения полей запроса.

Административный kubeconfig предназначен для подготовки стенда. prepare_mcp.py
создаёт data/cluster/agent-kubeconfig.yaml с отдельным токеном ServiceAccount agent
для MCP. Чтение и изменения выполняются через этот ServiceAccount.
Само наличие Role не ограничивает административные credentials. Подробнее —
[подключение инструментов](agent.md).

## Остановка

```bash
bash scripts/cluster.sh stop
```

Останавливает контейнер ноды с сохранением состояния. Для продолжения — up.
Docker Desktop после stop остаётся запущен.

## Объём реализации

Поднят базовый здоровый api с двумя репликами, probes, Service и ConfigMap.
18 сценариев из evaluation/scenarios.json реализованы в src/benchmark/fixtures.py.
Перед каждой задачей восстанавливаются тестовые нагрузки agent-lab. Подготовка
проверяет предусловия, сохраняет исходные спецификации и Pod UID. После прогона
восстанавливается базовый api. Образ evaluation/fixture_app обеспечивает повторяющиеся
логи конфигурации, управляемую readiness и ошибку обязательной переменной окружения.
Сценарий Pending резервирует CPU до создания batch; недоступный образ проверяется
с imagePullPolicy=Never без обращения к внешнему registry.

Основания конфигурации: [файл конфигурации k3d](https://k3d.io/stable/usage/configfile/),
[отдельный kubeconfig](https://k3d.io/stable/usage/kubeconfig/).
