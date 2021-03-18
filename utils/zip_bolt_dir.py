import shutil
import turibolt as bolt
import os

if __name__ == '__main__':
    results_folder = os.path.join(bolt.ARTIFACT_DIR, 'results')
    date_folder = os.path.join(results_folder, os.listdir(results_folder)[0])
    job_id = os.listdir(date_folder)[0]
    shutil.make_archive(os.path.join(bolt.ARTIFACT_DIR, 'results_zip'), 'zip', date_folder)
