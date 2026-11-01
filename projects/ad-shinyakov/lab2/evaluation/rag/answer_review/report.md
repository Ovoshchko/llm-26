# Сквозная проверка пункта 6

LLM из первой лабораторной: `glm4:9b` через Ollama.
Коллекция: `eval_config_4f81592daeca8af76b57`. Использован настоящий Qwen-эмбеддер и Qdrant.
Проверка по HTTP API: вопрос → вектор → поиск → контекст → LLM → ответ с источниками.

| Сценарий | top-k | HTTP | Проверка | Время, с |
|---|---:|---:|---|---:|
| single_hop | 3 | 200 | True | 28.0 |
| multi_hop | 5 | 200 | False | 100.7 |
| out_of_scope | 3 | 200 | True | 0.9 |

Все сниппеты и URL сверены с проверенным embedding bundle.
Для multi-hop дополнительно проверены оба DNS-имени и цитирование минимум двух документов.
Это три smoke-сценария, не полная оценка качества QA на 20 вопросах.

## single_hop

В Service заданы port: 80 и targetPort: 9376. На какой порт обращается клиент и на какой порт Pod попадёт трафик?

Клиент обращается на порт 80, а трафик попадает на порт 9376 в Pod.

Источники: [1] [2]

- [1] [Service > Defining a Service > Port definitions {#field-spec-ports}](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/service/)
- [2] [Service > Defining a Service](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/service/)

## multi_hop

Есть StatefulSet db с serviceName=db-headless в namespace prod, обычной нумерацией с нуля и доменом cluster.local. Headless Service имеет именованный TCP-порт sql:5432. Какое полное DNS-имя будет у первого Pod и какое имя SRV-запроса использовать для sql?

The full DNS name for the first Pod will be db-0.prod.svc.cluster.local, and the SRV query to use for sql will be _sql._tcp.db.prod.svc.cluster.local.

Источники: [1] [2]

- [1] [StatefulSets > Pod Identity > Stable Network ID](https://v1-35.docs.kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [2] [DNS for Services and Pods > Services > SRV records](https://v1-35.docs.kubernetes.io/docs/concepts/services-networking/dns-pod-service/)

## out_of_scope

Как приготовить шарлотку с яблоками?

В найденной документации недостаточно информации для подтверждённого ответа. Уточните вопрос или расширьте корпус.


Полные ответы, сниппеты, ID, время, usage и настройки: [results.json](results.json).
