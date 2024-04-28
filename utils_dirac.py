from qci_client import QciClient
import os
from dateutil import parser


def sample_qubo(Q, num_samples):
    qubo = Q + Q.T
    qubo = qubo/2

    api_token = API_TOKEN
    api_url = API_URL
    qci_token = QCI_TOKEN
    qci_url = QCI_URL

    client = QciClient(api_token=api_token, url=api_url)
    file_def = {"file_name": "QKP qubo", 
                "file_config": {"qubo": {"data": qubo}}}
    file_id = client.upload_file(file_def)["file_id"]
    job_tags = ["QKP", "example", "binary"]
    job_body = client.build_job_body(job_type="sample-qubo", qubo_file_id=file_id, 
                        job_params={"sampler_type": "dirac-1", "nsamples": num_samples}, job_tags=job_tags)
    response = client.process_job(job_type="sample-qubo", job_body=job_body, wait=True)
    return response

def wes_run_qubo(Q, num_samples):
    response = sample_qubo(Q, num_samples)
    solution = response['results']['file_config']['quadratic_unconstrained_binary_optimization_results']['solutions'][0]
    start = response['job_info']['job_status']['running_at_rfc3339nano']
    end = response['job_info']['job_status']['completed_at_rfc3339nano']

    #transform the time to seconds
    start_dt = parser.parse(start)
    end_dt = parser.parse(end)

    print("Time Difference: ", (end_dt - start_dt).total_seconds())
    runtime = (end_dt - start_dt).total_seconds()
        
    return solution, runtime