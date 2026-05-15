properties([disableConcurrentBuilds()])

node('slave-xg') {

    withEnv(["SCRIPTS_FOLDER=${env.WORKSPACE}/.jenkins","CI_ENV=true"]) {
        stage("Init")  {

            deleteDir()
            checkout scm
            sh "chmod +x -R $SCRIPTS_FOLDER"
        }
        stage ("Build"){
            sh "$SCRIPTS_FOLDER/printLog.sh 'INFO' 'Jenkins-Build' 'Starting Python installation through poetry'"
            sh "poetry install --no-interaction"
            sh "$SCRIPTS_FOLDER/printLog.sh 'INFO' 'Jenkins-Build' 'Finishing Python installation through poetry'"
        }

        try {
        stage('Deploy') {
            sh '''
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Deploying Ollama container."
                docker compose up ollama-gpu --detach
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Downloading models."
                .jenkins/downloadModels.sh
            '''
        }

        stage('Generate Input') {
            sh '''
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Generating per-repo test case JSONs from GitHub repositories."
                poetry run python ril2m/input/fetch_testcases.py
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Test case generation complete (see summary above)."
            '''
            archiveArtifacts artifacts: 'ril2m/input/context/*.json', allowEmptyArchive: true
        }

        stage('Cross-Validation') {
            sh '''
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Running leave-one-out cross-validation (embedding + LLM query per fold)."
                poetry run python ril2m/crossvalidation.py
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Cross-validation complete (see summary above)."
            '''
        }

        stage('Compute Metrics') {
            sh '''
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Computing M1-M8 evaluation metrics."
                poetry run python ril2m/metrics.py
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Metrics complete (see summary above)."
            '''
        }

        }//EndTry
        finally {
        stage('Tear-down') {
            sh '''
                $SCRIPTS_FOLDER/printLog.sh "INFO" "Jenkinsfile" "Tear-down Ollama container."
                $SCRIPTS_FOLDER/storeContainerLogs.sh
                docker compose down ollama-gpu
            '''
        }
        stage('Archive results') {
            // Always archive run outputs — even when Cross-Validation fails mid-run
            // so partial results (already-saved .txt files) are not lost.
            archiveArtifacts artifacts: 'ril2m/chroma_db/**/manifest.json', allowEmptyArchive: true
            archiveArtifacts artifacts: 'outputs/crossval/**/*', allowEmptyArchive: true
            archiveArtifacts artifacts: 'outputs/metrics/**/*', allowEmptyArchive: true
        }
        stage('Archive logs') {
            archiveArtifacts artifacts: 'logs/**/*', allowEmptyArchive: true
        }
    }

    }//EndWithEnv
}//EndNodeXGPU