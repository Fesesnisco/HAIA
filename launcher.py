import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Sesión de SageMaker
sagemaker_session = sagemaker.Session()

# Rol de ejecución (toma el rol de la instancia actual)
role = sagemaker.get_execution_role()

# Bucket de S3 donde tengas el genetico
# bucket = sagemaker_session.default_bucket()

sklearn_estimator = SKLearn(
    entry_point="pruebas.py",  # El script de entrenamiento
    source_dir=".",  # Donde está el código
    # source_dir="s3://sagemaker-jmi/".format(bucket),  # Esto sería si tienes el código en S3
    role=role,
    instance_type="ml.t3.large",  # Tipo de instancia (puedes cambiarlo) medium no va
    instance_count=1,
    framework_version="0.23-1",  # Versión de SKLearn
    py_version="py3",
    sagemaker_session=sagemaker_session,
)

sklearn_estimator.fit()
