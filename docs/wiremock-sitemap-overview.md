# WireMock Docs Sitemap Overview

Source: `https://wiremock.org/sitemap-0.xml`  
Local copy: `data/sitemap.xml`  
Total /docs/ pages: **76**

---

## Structure

The sitemap organises WireMock documentation into flat top-level pages and several nested sections. Depth here means URL path depth below `/docs/`.

### Top-level pages (depth 1)

| Page | URL |
|---|---|
| Overview | /docs/overview/ |
| v4 Beta | /docs/v4/ |
| Download and Install | /docs/download-and-installation/ |
| Getting Started / Tutorials | /docs/getting-started/ |
| FAQ | /docs/faq/ |
| Java Usage | /docs/java-usage/ |
| Configuration | /docs/configuration/ |
| Running without HTTP Server | /docs/running-without-http-server/ |
| Jetty 12 | /docs/jetty-12/ |
| JUnit Jupiter | /docs/junit-jupiter/ |
| JUnit Extensions | /docs/junit-extensions/ |
| Stubbing | /docs/stubbing/ |
| Request Matching | /docs/request-matching/ |
| Response Templating | /docs/response-templating/ |
| Faker Extension | /docs/faker-extension/ |
| Simulating Faults | /docs/simulating-faults/ |
| Stateful Behaviour | /docs/stateful-behaviour/ |
| Proxying | /docs/proxying/ |
| Verifying | /docs/verifying/ |
| Record and Playback | /docs/record-playback/ |
| WebSockets | /docs/websockets/ |
| Webhooks and Callbacks | /docs/webhooks-and-callbacks/ |
| GraphQL | /docs/graphql/ |
| gRPC | /docs/grpc/ |
| JWT | /docs/jwt/ |
| HTTPS | /docs/https/ |
| Spring Boot | /docs/spring-boot/ |
| Multi-domain Mocking | /docs/multi-domain-mocking/ |
| Extending WireMock | /docs/extending-wiremock/ |
| Standalone | /docs/standalone/ |
| Support | /docs/support/ |
| Commercial | /docs/commercial/ |
| External Resources | /docs/external-resources/ |
| WireMock Cloud | /docs/wiremock-cloud/ |

---

### Nested sections (depth 2)

#### /docs/advanced/ — Legacy / Advanced Topics
- /docs/advanced/deploy-to-servlet-container/
- /docs/advanced/java7/

#### /docs/extensibility/ — Extending WireMock
- /docs/extensibility/filtering-requests/
- /docs/extensibility/transforming-responses/
- /docs/extensibility/custom-matching/
- /docs/extensibility/listening-for-serve-events/
- /docs/extensibility/listening-for-settings-changes/
- /docs/extensibility/listening-for-stub-changes/
- /docs/extensibility/admin-api-extensions/
- /docs/extensibility/adding-template-helpers/
- /docs/extensibility/adding-template-model-data/
- /docs/extensibility/adding-mappings-loader/
- /docs/extensibility/stub-metadata/

#### /docs/messaging/ — Messaging Framework
- /docs/messaging/overview/
- /docs/messaging/websockets/
- /docs/messaging/stubbing/
- /docs/messaging/verification/
- /docs/messaging/sending-messages/

#### /docs/quickstart/
- /docs/quickstart/java-junit/

#### /docs/solutions/ — Language & Framework Integrations
- /docs/solutions/ai/
- /docs/solutions/android/
- /docs/solutions/c_cpp/
- /docs/solutions/dotnet/
- /docs/solutions/golang/
- /docs/solutions/graphql/
- /docs/solutions/groovy/
- /docs/solutions/jvm/
- /docs/solutions/kotlin/
- /docs/solutions/kubernetes/
- /docs/solutions/nodejs/
- /docs/solutions/pact/
- /docs/solutions/python/
- /docs/solutions/quarkus/
- /docs/solutions/rust/
- /docs/solutions/service-virtualization/
- /docs/solutions/spring-boot-integration/
- /docs/solutions/testcontainers/

#### /docs/standalone/ — Running Standalone
- /docs/standalone/admin-api-reference/
- /docs/standalone/administration/
- /docs/standalone/docker/
- /docs/standalone/java-jar/

---

## Notes

- The sitemap does **not** include `/docs/` itself as a proper content page — it acts as the index/landing page.
- Some pages visible in the sidebar nav (e.g. `/docs/solutions/*`) are in the sitemap but were not returned by `SitemapLoader` with a `/docs/` filter — confirmed by `experiment.py` Probe B.
- The sitemap is fetched from `/sitemap-0.xml` (not `/sitemap.xml`, which returns 404).
