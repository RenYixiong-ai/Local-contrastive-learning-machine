# Description

This project aims to reproduce the results from Xie, M., Wang, Y., & Huang, H.  

[Local-contrastive-learning machine with both generalization and adversarial robustness: A statistical physics analysis. Sci. China Phys. Mech. Astron. 68, 210511 (2025).](https://doi.org/10.1007/s11433-024-2504-8)  

Official code repository: [https://github.com/PMI-SYSU/FBM](https://github.com/PMI-SYSU/FBM)  


# File Structure

* **utils.py**  
   - Contains three sections: random seed setting, dataset loading, and dataset processing.  
* **train**  
   - Provides the training function for FBM, which returns a trained single-layer fully connected network.  
* **models**  
   - Includes the construction of FBM and MLP neural networks.  
* **loss**  
   - Calculates the Fermi-Bose loss.  
* **exampe**  
   - Reaserch details.

# Run

```python
python main.py
```
