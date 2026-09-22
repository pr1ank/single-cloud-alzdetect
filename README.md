# Single-Cloud Alzheimer's Disease Analysis & Cloud Efficiency Platform

1. Install Python **3.11 or 3.12**.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

The first launch downloads public MRI data and EfficientNetB0 weights, trains automatically, runs five scalability experiments, and saves results. CPU initialization can take several minutes or longer. Later launches reuse completed experiments. No accounts, credentials, cloud services, configuration, or uploads are required. Internet failures trigger clearly labeled synthetic demo data; unavailable pretrained weights are disclosed. Research and educational use only. This system is not a medical diagnostic device. See the application's Methodology page and `assets/METHODOLOGY.md` for sources, assumptions, and limitations.
