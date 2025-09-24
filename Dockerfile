FROM continuumio/miniconda3

# Set environment variables
ENV ENV_NAME=you-have-to-be-realistic
ENV PYTHON_VERSION=3.9

# Set working directory
WORKDIR /workspace
COPY . /workspace

# Install all packages and create the environment
RUN conda create -y -n $ENV_NAME python=$PYTHON_VERSION && \
    conda run -n $ENV_NAME conda install -y \
        h5py \
        matplotlib \
        numpy \
        scikit-learn \
        scipy \
        jupyter \
        tqdm=4.63.0 && \
    conda run -n $ENV_NAME pip install tensorflow==2.15.0.post1 && \
    conda run -n $ENV_NAME python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)" && \
    conda clean -afy

# Expose the Jupyter Notebook port
EXPOSE 8888

# Default command to run Jupyter Notebook inside the conda env
CMD ["conda", "run", "-n", "you-have-to-be-realistic", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--notebook-dir=/workspace"]