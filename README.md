# iai-one-point-cloud
IAI March 2025 - Point Cloud Assignment

## Task 1 – Ground Level Estimation

### Objective
The goal of Task 1 is to estimate the ground level from the LiDAR point cloud data.

### Method
The ground level was estimated using the histogram of the `z` values in the point cloud.

Steps followed:
1. Extract the `z` coordinate from all points in the dataset.
2. Generate a histogram of the `z` values.
3. Identify the histogram bin with the highest frequency.
4. Use the center of that bin as the estimated ground level.

This approach is based on the assumption that the ground forms the densest horizontal layer in the point cloud, so it appears as the strongest peak in the `z`-value distribution.

A small margin above the estimated ground level was then used to keep only the points above ground for the next tasks.

---

### Dataset 1

**Estimated ground level:** 61.282

#### Histogram Plot
![Dataset 1 Histogram](images/dataset1_histogram.png)

#### Observation
For dataset 1, the histogram showed a clear dominant peak in the lower `z` range, which was selected as the ground level.  
This value was then used to remove the ground points before clustering.

---

### Dataset 2

**Estimated ground level:** 61.299

#### Histogram Plot
![Dataset 2 Histogram](images/dataset2_histogram.png)

#### Observation
For dataset 2, the histogram also showed a strong concentration of points at the lower `z` range.  
The center of the densest bin was selected as the estimated ground level and used for above-ground point extraction.

---

### Summary
The histogram-based method provided a simple and effective way to estimate the ground level for both datasets.  
The estimated ground level was then used as the threshold for separating ground points from above-ground structures such as trees, poles, and catenary-related points.
