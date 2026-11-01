# Вопросы и эталонные ответы

Корпус: Kubernetes Concepts, release-1.35. 10 single-hop и 10 multi-hop.

Полная разметка с точными цитатами: [questions.json](questions.json).
Привязка цитат к чанкам: [labels/3cccc393ec5ee49d.json](labels/3cccc393ec5ee49d.json).

## q01 · single_hop

В Service заданы port: 80 и targetPort: 9376. На какой порт обращается клиент, на какой порт Pod попадёт трафик и что будет, если targetPort не указать?

**Эталон:** Клиент обращается к порту 80 Service, а Service направляет трафик на TCP-порт 9376 выбранных Pod. Если targetPort не задан, он по умолчанию равен port, то есть в этом примере 80.

**Основания ответа:**

- Прочитать пример Service и правило значения targetPort по умолчанию.

**Источники:**

- [service.md](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/service/)

## q02 · single_hop

Приложение долго запускается. Чем различаются последствия провала livenessProbe и readinessProbe и как startupProbe защищает период запуска?

**Эталон:** При провале livenessProbe kubelet останавливает контейнер, после чего действует restartPolicy. Провал readinessProbe означает, что контейнер не готов обслуживать запросы; это влияет на включение Pod в бэкенды Service, а не само по себе вызывает перезапуск. Пока startupProbe не успешна, остальные probes отключены; её провал приводит к остановке контейнера с последующим применением restartPolicy.

**Основания ответа:**

- Сопоставить три вида probes в одном разделе Pod Lifecycle.

**Источники:**

- [pod-lifecycle.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)

## q03 · single_hop

Обновил ConfigMap, но файл в контейнере не изменился. Чем обычное монтирование ConfigMap в volume отличается от монтирования через subPath?

**Эталон:** При обычном монтировании ConfigMap в volume проецируемые ключи обновляются со временем, а не обязательно немедленно: есть период синхронизации kubelet и задержка кеша. Монтирование через subPath не получает обновления ConfigMap.

**Основания ответа:**

- Проверить правила обновления проецируемых файлов и исключение subPath.

**Источники:**

- [configmap.md](https://v1-35.docs.kubernetes.io/docs/concepts/configuration/configmap/)

## q04 · single_hop

Как записать пароль в Secret без ручного base64-кодирования? Какое значение выберется, если одинаковый ключ задан в data и stringData?

**Эталон:** Для обычных строк можно использовать stringData; значения data должны быть base64-кодированными. stringData объединяется с data, и при совпадении ключа приоритет имеет stringData.

**Основания ответа:**

- Прочитать ограничения data/stringData и правило приоритета ключей.

**Источники:**

- [secret.md](https://v1-35.docs.kubernetes.io/docs/concepts/configuration/secret/)

## q05 · single_hop

Для чего Kubernetes использует requests и limits? Может ли контейнер использовать больше CPU или памяти, чем указано в request?

**Эталон:** Scheduler использует requests при выборе узла, а limits ограничивают потребление при выполнении. При наличии свободных ресурсов контейнер может использовать больше своего request; request не является верхним пределом. CPU limit обеспечивается throttling, а memory limit — реактивными OOM kills при давлении на память, поэтому превышение памяти не обязательно приводит к немедленному завершению.

**Основания ответа:**

- Различить назначение requests и limits и механизмы ограничения CPU и памяти.

**Источники:**

- [manage-resources-containers.md](https://v1-35.docs.kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

## q06 · single_hop

Что ограничивают maxUnavailable и maxSurge при RollingUpdate Deployment и как округляются проценты в этих параметрах?

**Эталон:** maxUnavailable ограничивает число недоступных Pod относительно желаемого числа реплик; процент округляется вниз. maxSurge разрешает дополнительные Pod сверх желаемого числа; процент округляется вверх. По умолчанию оба равны 25%; одновременно задавать оба равными нулю нельзя. Завершающиеся Pod могут временно увеличивать фактическое число существующих Pod.

**Основания ответа:**

- Прочитать определения и округления параметров RollingUpdate, учесть terminating Pods.

**Источники:**

- [deployment.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/)

## q07 · single_hop

Pod в namespace test не находит Service data из namespace prod по короткому имени data. Какое DNS-имя нужно использовать и почему?

**Эталон:** Короткое имя ищется прежде всего в namespace самого Pod. Для Service из prod нужно указать namespace: data.prod; полное имя при домене cluster.local — data.prod.svc.cluster.local.

**Основания ответа:**

- Проверить namespace в DNS search list и правила обращения к Service другого namespace.

**Источники:**

- [dns-pod-service.md](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/dns-pod-service/)

## q08 · single_hop

Почему основной контейнер не стартует, пока обычные init containers не завершились? Можно ли обычному init container добавить readinessProbe?

**Эталон:** Обычные init containers выполняются последовательно; каждый должен успешно завершиться до следующего, а приложение запускается после завершения всех. Обычные init containers не поддерживают readinessProbe (как и lifecycle, livenessProbe, startupProbe). Это ограничение не следует переносить на native sidecar containers.

**Основания ответа:**

- Прочитать порядок обычных init containers и отличие от sidecar.

**Источники:**

- [init-containers.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/pods/init-containers/)

## q09 · single_hop

Если несколько NetworkPolicy выбирают один Pod, какая из них имеет приоритет? Достаточно ли разрешить ingress на получателе, когда отправитель изолирован по egress?

**Эталон:** Политики аддитивны: разрешения объединяются, порядок и приоритет не определяют результат. Для соединения должны разрешать трафик обе стороны: egress отправителя и ingress получателя. Одного разрешающего ingress недостаточно, если egress запрещает соединение. Предполагается, что сетевой плагин поддерживает NetworkPolicy.

**Основания ответа:**

- Проверить объединение политик и условия разрешения соединения между двумя Pod.

**Источники:**

- [network-policies.md](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/network-policies/)

## q10 · single_hop

Гарантирует ли ReadWriteOnce, что PVC использует только один Pod? Какой режим нужен, если доступ на запись должен иметь ровно один Pod во всём кластере?

**Эталон:** ReadWriteOnce ограничивает read-write доступ одним узлом, но допускает несколько Pod на этом узле. Для ограничения одним Pod нужен ReadWriteOncePod; этот режим поддерживается CSI volumes и требует совместимых версий компонентов.

**Основания ответа:**

- Различить ограничения по узлу и по Pod в Access Modes.

**Источники:**

- [persistent-volumes.md](https://v1-35.docs.kubernetes.io/docs/concepts/storage/persistent-volumes/)

## q11 · multi_hop

Deployment передаёт настройки из ConfigMap через env. Я изменил только ConfigMap и затем увеличил replicas, но в уже работающих контейнерах старые значения. Почему и как обновить все реплики управляемым rollout?

**Эталон:** Переменные окружения из ConfigMap не обновляются автоматически — контейнерам нужен перезапуск. Изменение только ConfigMap не меняет Pod template Deployment, а изменение replicas само по себе не запускает rollout. Для управляемого обновления всех реплик нужно изменить spec.template, например annotation с версией/хешем конфигурации, чтобы Deployment создал заменяющие Pod с новыми значениями. Новые Pod при масштабировании не обновляют окружение уже существующих.

**Основания ответа:**

- Установить, что env из ConfigMap фиксируется при запуске.
- Найти условие запуска rollout у Deployment.
- Связать обновление конфигурации с изменением Pod template, а не только replicas.

**Источники:**

- [configmap.md](https://v1-35.docs.kubernetes.io/docs/concepts/configuration/configmap/)
- [deployment.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/)

## q12 · multi_hop

Контейнер остаётся Running и проходит livenessProbe, но readinessProbe падает. Как это связано с serving и ready в EndpointSlice и почему publishNotReadyAddresses=true меняет трактовку ready?

**Эталон:** Успешная liveness не заменяет readiness: провал readiness делает Pod непригодным для обычного трафика Service без обязательного перезапуска контейнера. У endpoint, связанного с Pod, serving соответствует Pod Ready; ready обычно означает serving и отсутствие terminating. Поэтому при неготовом, незавершающемся Pod обычно serving=false и ready=false. Для Service с publishNotReadyAddresses=true условие ready принудительно true и больше не доказывает готовность приложения.

**Основания ответа:**

- Различить liveness и readiness.
- Связать Pod readiness с serving endpoint.
- Применить определение ready и исключение publishNotReadyAddresses.

**Источники:**

- [pod-lifecycle.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [endpoint-slices.md](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/endpoint-slices/)

## q13 · multi_hop

Есть StatefulSet db с serviceName=db-headless в namespace prod, обычной нумерацией с нуля и доменом cluster.local. Headless Service имеет именованный TCP-порт sql:5432. Какое имя будет у первого Pod и что вернёт SRV-запрос для sql?

**Эталон:** Первый Pod называется db-0, его полное DNS-имя — db-0.db-headless.prod.svc.cluster.local. SRV-запрос _sql._tcp.db-headless.prod.svc.cluster.local для headless Service возвращает ответы по обслуживающим Pod с номером порта 5432 и DNS-именами этих Pod, включая db-0.db-headless.prod.svc.cluster.local, когда Pod опубликован в DNS. Предполагается, что Headless Service создан и выбирает эти Pod.

**Основания ответа:**

- Вывести hostname и домен Pod из правил StatefulSet.
- Найти формат SRV для именованного порта headless Service.
- Подставить имя, namespace, домен и порт из условия.

**Источники:**

- [statefulset.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [dns-pod-service.md](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/dns-pod-service/)

## q14 · multi_hop

У StatefulSet не задана persistentVolumeClaimRetentionPolicy, а у его PV стоит reclaimPolicy: Delete и CSI-драйвер поддерживает удаление. Удаление StatefulSet сразу удалит данные? Что изменится, если затем вручную удалить оставшийся PVC?

**Эталон:** По умолчанию StatefulSet использует Retain для PVC: удаление самого StatefulSet не удаляет автоматически его claims, поэтому Delete у PV сам по себе ещё не запускает удаление хранилища. После удаления и освобождения PVC reclaimPolicy Delete позволяет удалить PV и соответствующий storage asset. Это разные уровни политики; обработка асинхронная и учитывает защиту используемых объектов.

**Основания ответа:**

- Установить политику сохранения PVC у StatefulSet по умолчанию.
- Проследить освобождение claim до reclaim policy PV.
- Различить удаление StatefulSet и явное удаление PVC.

**Источники:**

- [statefulset.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [persistent-volumes.md](https://v1-35.docs.kubernetes.io/docs/concepts/storage/persistent-volumes/)

## q15 · multi_hop

Pod использует локальный PV с nodeAffinity на зону zone-a, но сам Pod требует zone-b через nodeSelector. Может ли режим WaitForFirstConsumer сам устранить конфликт, если ограничения Pod и существующего PV не менять?

**Эталон:** Нет. Pod, использующий PV, может быть запланирован только на узлы, разрешённые nodeAffinity PV; одновременно должны выполняться ограничения самого Pod. WaitForFirstConsumer откладывает выбор/создание и привязку тома до появления потребителя и учитывает его ограничения, но сам не делает существующий локальный том доступным в другой зоне. При непересекающихся требованиях нужны изменения Pod, выбора тома или размещения данных.

**Основания ответа:**

- Установить ограничение узлов для Pod, использующего PV.
- Проверить, что WaitForFirstConsumer меняет момент привязки и учитывает ограничения Pod.
- Определить отсутствие совместимого размещения при неизменных конфликтующих требованиях.

**Источники:**

- [persistent-volumes.md](https://v1-35.docs.kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [storage-classes.md](https://v1-35.docs.kubernetes.io/docs/concepts/storage/storage-classes/)

## q16 · multi_hop

В каждом Pod один контейнер. В исходном YAML не указан CPU request, но LimitRange при создании подставляет defaultRequest.cpu=100m. HPA использует averageUtilization=50%. Есть две готовые реплики, каждая потребляет 100m CPU. Какой utilization и desiredReplicas получатся без пропущенных метрик, ограничений min/max и стабилизации?

**Эталон:** После admission у каждого контейнера request=100m благодаря LimitRange. Потребление 100m означает 100% utilization относительно request. Отношение текущего utilization к целевому равно 100/50=2, поэтому desiredReplicas=ceil(2×2)=4. Отсутствие request в исходном YAML не означает, что request отсутствует в созданном Pod.

**Основания ответа:**

- Найти автоматическую подстановку request через LimitRange.
- Определить знаменатель CPU utilization в HPA.
- Применить формулу ceil(currentReplicas × current/target).

**Источники:**

- [limit-range.md](https://v1-35.docs.kubernetes.io/docs/concepts/policy/limit-range/)
- [horizontal-pod-autoscale.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)

## q17 · multi_hop

Узел пометили unschedulable командой cordon. Почему обычные новые Pod на него не назначаются, существующие продолжают работать, а новый Pod DaemonSet всё же может появиться?

**Эталон:** Cordon запрещает scheduler размещать новые обычные Pod, но не выселяет уже работающие. Контроллер DaemonSet автоматически добавляет toleration node.kubernetes.io/unschedulable:NoSchedule, поэтому DaemonSet Pod может быть назначен на такой узел при выполнении остальных ограничений.

**Основания ответа:**

- Определить влияние cordon на новые и существующие Pod.
- Найти автоматически добавляемую toleration DaemonSet.
- Объяснить исключение для DaemonSet без предположения, что cordon выселяет Pod.

**Источники:**

- [nodes.md](https://v1-35.docs.kubernetes.io/docs/concepts/architecture/nodes/)
- [daemonset.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/daemonset/)

## q18 · multi_hop

У Deployment три Ready-реплики, PDB требует minAvailable=3, а RollingUpdate настроен maxUnavailable=1 и maxSurge=0. Одинаково ли PDB ограничит kubectl drain и rollout этого Deployment?

**Эталон:** Нет. Drain использует Eviction API и должен учитывать PDB: при трёх Ready и minAvailable=3 добровольное выселение одной реплики не допускается, пока не появится бюджет. RollingUpdate самого Deployment не ограничивается PDB, хотя недоступные Pod учитываются при расчёте бюджета для eviction. При maxUnavailable=1 и maxSurge=0 rollout может сначала снизить доступность до двух реплик; его предел задаётся стратегией Deployment.

**Основания ответа:**

- Установить применение PDB к Eviction API/drain.
- Найти исключение для rolling updates контроллера.
- Применить maxUnavailable Deployment к заданному числу реплик.

**Источники:**

- [disruptions.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/pods/disruptions/)
- [deployment.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/)

## q19 · multi_hop

Хочу удалять завершённые Jobs моего CronJob примерно через час, а не только хранить последние несколько запусков. Где задать TTL, от какого события он отсчитывается и могут ли history limits удалить Job раньше?

**Эталон:** Для новых Jobs задайте spec.jobTemplate.spec.ttlSecondsAfterFinished=3600 в CronJob: jobTemplate задаёт создаваемые Jobs. TTL отсчитывается от перехода Job в Complete или Failed, а не от запуска. successfulJobsHistoryLimit/failedJobsHistoryLimit ограничивают количество сохранённых Jobs независимо от TTL и могут удалить старые раньше часа. TTL означает право на последующую очистку контроллером, а не точную гарантию времени удаления или минимального срока хранения.

**Основания ответа:**

- Перейти от CronJob к спецификации создаваемого Job через jobTemplate.
- Найти TTL finished Job и момент начала отсчёта.
- Сопоставить временной TTL с количественным ограничением истории.

**Источники:**

- [cron-jobs.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [ttlafterfinished.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/ttlafterfinished/)

## q20 · multi_hop

Pod удаляется, но приложение ещё завершает запросы, а рядом работает native sidecar-прокси. Почему sidecar не останавливают сразу и что в этот период означают serving, terminating и ready в EndpointSlice?

**Эталон:** Kubelet откладывает остановку native sidecars до полного завершения основных контейнеров, затем завершает sidecars в обратном порядке их объявления. Endpoint становится terminating после получения Pod deletion timestamp, до фактического выхода контейнеров; serving может ещё быть true, если Pod остаётся Ready. Обычно ready=false для terminating endpoint, за исключением publishNotReadyAddresses=true. Прокси обычно игнорирует terminating endpoints, но может использовать serving+terminating, если все доступные endpoints завершаются. Поэтому удаление Pod не тождественно мгновенному завершению приложения и обслуживающего sidecar.

**Основания ответа:**

- Установить порядок остановки native sidecars.
- Связать удаление Pod с terminating endpoint, а не с мгновенным выходом процессов.
- Различить serving и ready и учесть исключения маршрутизации.

**Источники:**

- [sidecar-containers.md](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/)
- [endpoint-slices.md](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/endpoint-slices/)
