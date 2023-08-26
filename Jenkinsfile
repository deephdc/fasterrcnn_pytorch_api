#!/usr/bin/groovy

@Library(['github.com/indigo-dc/jenkins-pipeline-library@1.4.0']) _

def job_result_url = ''

ci_cd_image = 'mteamkit/cicd-fasterrcnn_pytorch_api'

pipeline {
    agent {
        docker { image "${ci_cd_image}" }
    }

    environment {
        author_name = "Fahimeh"
        author_email = "f.alibabaee@gmail.com"
        app_name = "fasterrcnn_pytorch_api"
        job_location = "Pipeline-as-code/DEEP-OC-org/DEEP-OC-fasterrcnn_pytorch_api/${env.BRANCH_NAME}"
    }

    stages {
        stage('Code fetching') {
            steps {
                checkout scm
            }
        }

        stage('Style analysis: PEP8') {
            steps {
                ToxEnvRun('qc.sty')
            }
        }

        stage('Unit testing with Coverage') {
            steps {
                ToxEnvRun('qc.cov')
            }
            post {
                success {
                    HTMLReport('htmlcov', 'index.html', 'Coverage report')
                    CoberturaReport('**/coverage.xml')
                }
            }
        }

        stage('Security scanner') {
            steps {
                ToxEnvRun('qc.sec')
                script {
                    if (currentBuild.result == 'FAILURE') {
                        currentBuild.result = 'UNSTABLE'
                    }
               }
            }
            post {
               always {
                    HTMLReport("/tmp/bandit", 'index.html', 'Bandit report')
                }
            }
        }

        stage('Metrics gathering') {
            agent {
                label 'sloc'
            }
            steps {
                checkout scm
                SLOCRun()
            }
            post {
                success {
                    SLOCPublish()
                }
            }
        }

        stage("Re-build Docker images") {
            when {
                anyOf {
                   branch 'master'
                   branch 'test'
                   buildingTag()
               }
            }
            steps {
                script {
                    def job_result = JenkinsBuildJob("${env.job_location}")
                    job_result_url = job_result.absoluteUrl
                }
            }
        }

}
}
