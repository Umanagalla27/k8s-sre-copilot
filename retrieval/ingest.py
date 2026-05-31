import os
import json
import logging
from qdrant_client.http import models
from storage.qdrant import get_qdrant_client, COLLECTION_NAME, init_qdrant
from retrieval.embed import get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval.ingest")

DOCUMENTS = [
    {
        "id": 1,
        "title": "Troubleshooting CrashLoopBackOff Pods",
        "category": "Runbook",
        "content": "A CrashLoopBackOff status indicates a pod is repeatedly starting and failing. To troubleshoot: 1. Run 'kubectl logs <pod-name> --previous' to inspect logs from the last termination. 2. Describe the pod using 'kubectl describe pod <pod-name>' and check the 'Events' section. 3. Check for application configuration issues, missing secrets/configmaps, or incorrect entrypoint/cmd commands. Common causes include exit code 1 (general error) and exit code 137 (SIGKILL)."
    },
    {
        "id": 2,
        "title": "Resolving Pod OOMKilled (Exit Code 137)",
        "category": "Runbook",
        "content": "OOMKilled means the container was terminated by the Linux kernel Out-Of-Memory killer because it exceeded its memory limit. 1. Identify the container using 'kubectl describe pod' and locate 'Last State: Terminated' with 'Reason: OOMKilled'. 2. Increase the memory limit in the pod specification under 'resources.limits.memory'. 3. Profile the application for memory leaks. Avoid setting limits too low for JVM applications, which need extra memory headroom above heap size."
    },
    {
        "id": 3,
        "title": "Debugging CoreDNS Loopback and Resolution Failures",
        "category": "Runbook",
        "content": "CoreDNS loops happen when the query sent to CoreDNS resolves back to CoreDNS itself, causing a loop. Check the CoreDNS ConfigMap for the 'loop' plugin. If loops are detected, CoreDNS will crash. Remove the 'loop' directive or modify the node's /etc/resolv.conf. Use 'kubectl exec -it <test-pod> -- nslookup kubernetes.default' to verify internal cluster DNS routing. Common symptoms are DNS timeout errors on service discovery."
    },
    {
        "id": 4,
        "title": "Managing PodDisruptionBudget (PDB) Blocking Evictions",
        "category": "Runbook",
        "content": "A PodDisruptionBudget (PDB) limits the number of pods of a replicated application that are down simultaneously. If node draining fails or hangs, check if a PDB is blocking evictions. Use 'kubectl get pdb -A' to check. If minAvailable is set to 100% or maxUnavailable is 0, evictions are permanently blocked during cluster upgrades. Temporarily delete or relax the PDB constraints to allow safe draining."
    },
    {
        "id": 5,
        "title": "Rolling Back Deployments using kubectl rollout undo",
        "category": "Runbook",
        "content": "To undo a failed deployment update, use 'kubectl rollout undo deployment/<deployment-name>'. You can rollback to a specific revision by adding '--to-revision=N'. View deployment history with 'kubectl rollout history deployment/<deployment-name>'. This is critical when bad configurations or bugged versions are rolled out to production and need immediate reversion."
    },
    {
        "id": 6,
        "title": "Debugging Ingress HTTP 502/503 Bad Gateway Errors",
        "category": "Runbook",
        "content": "HTTP 502 Bad Gateway and 503 Service Unavailable errors on ingress controllers (like ingress-nginx) mean the controller cannot communicate with the backend pod. 1. Check if backend endpoints exist using 'kubectl get endpoints <service-name>'. 2. Verify backend pod status is Running and probes (readiness/liveness) are passing. 3. Check if service port matches containerPort. 4. Verify network policies do not block traffic from the ingress namespace."
    },
    {
        "id": 7,
        "title": "Handling Node Disk Pressure and Evictions",
        "category": "Runbook",
        "content": "When a node runs out of disk space, kubelet flags it with 'DiskPressure' and starts evicting pods. 1. Run 'kubectl describe node <node-name>' to check conditions. 2. SSH into the node and run 'df -h' to identify space usage. 3. Clean unused container images via 'docker system prune -a' or 'crictl rmi --prune'. 4. Check for runaway log files under /var/log or docker container logging configurations."
    },
    {
        "id": 8,
        "title": "Resolving Pending PersistentVolumeClaim (PVC) Mounts",
        "category": "Runbook",
        "content": "A PVC remaining in Pending state indicates that the volume scheduler could not provision or bind a backing PersistentVolume. 1. Run 'kubectl describe pvc <pvc-name>' to find provisioning failure messages. 2. Verify the StorageClass exists and is spelled correctly. 3. For local volumes, check if the matching PV has the correct node affinity. 4. Ensure cloud provider permissions permit dynamic volume provisioning."
    },
    {
        "id": 9,
        "title": "Incident Postmortem: CoreDNS service loop failure in production",
        "category": "Postmortem",
        "content": "Severity: Critical. On 2026-04-12, the production API gateway experienced severe service resolution timeouts. The cause was a DNS loop triggered by a node update that modified /etc/resolv.conf nameservers. CoreDNS loop plugin detected recursion and self-terminated, cascading DNS failures across all microservices. Resolution: Removed the loop plugin from CoreDNS ConfigMap, restarted deployment, and updated node provisioning scripts to keep dnsmasq configurations clean."
    },
    {
        "id": 10,
        "title": "Incident Postmortem: Postgres database CPU exhaustion during peak traffic",
        "category": "Postmortem",
        "content": "Severity: High. On 2026-05-18, client checkout service failed due to DB timeouts. The primary Postgres database pod suffered 100% CPU exhaustion. RCA: An unindexed query on the 'incidents' table was called frequently. Resolution: Created composite index on (service, created_at) column, optimized the checkout query, and scaled the Postgres memory resources limit to 4Gi to avoid swapping."
    }
]

CORPUS_PATH = "./storage/runbooks_corpus.json"

def run_ingestion():
    # Make sure storage folder exists
    os.makedirs("./storage", exist_ok=True)
    
    # Save documents locally for BM25 reference
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(DOCUMENTS, f, indent=4)
    logger.info(f"Saved {len(DOCUMENTS)} runbooks to local corpus: {CORPUS_PATH}")

    # Initialize Qdrant Collection
    init_qdrant()

    # Generate Embeddings
    contents = [doc["content"] for doc in DOCUMENTS]
    embeddings = get_embeddings(contents)

    # Prepare Qdrant points
    points = []
    for i, doc in enumerate(DOCUMENTS):
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=embeddings[i],
                payload={
                    "title": doc["title"],
                    "category": doc["category"],
                    "content": doc["content"]
                }
            )
        )

    # Upsert points to Qdrant
    client = get_qdrant_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    logger.info(f"Successfully upserted {len(points)} documents into Qdrant collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    run_ingestion()
