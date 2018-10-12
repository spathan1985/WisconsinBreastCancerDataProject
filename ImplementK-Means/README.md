1. Use Dataset with imputed missing values from phase 1

2. Write code for ‘Initialization’ step for K value 2.
  * Steps:
    1. Choose any two points randomly from dataset as your initial means. 
    2. Out of input 699 datapoints, select any 2 random datapoints as your mean using numpy.random. 
    3. Since you only consider column A2 to A10 for your mean, this mean is nothing but nine dimensional vector. 
    4. Give first mean variable name as 𝜇2 and second mean variable name as 𝜇4. Here these two means represent two clusters.

3. Write code for ‘Assignment’ step
  * Steps: 
    1. You have defined two means in the previous step. 
    2. Now compute each of 699 datapoints euclidian distance from these two means. 
    3. So for each datapoint, you will have two distances. 
    4. Assign particular datapoint to cluster 2 if its distance from 𝜇2 is closer than its distance from 𝜇4 or vice versa.
    5. At the end of this step, you have every point assigned to any of two clusters.

4. Write code for ‘Recalculation’ step
  * So far, you have got every datapoint assigned to any of two clusters. 
  * Now in this step you will update means that you computed in first step.
  * Example: Let’s say after performing step b, you have 300 datapoints assigned to cluster 2 and remaining 399 datapoints 
  assigned to cluster 4. Update 𝜇2 by computing mean from these 300 datapoints and update 𝜇4 by computing mean from these 
  499 datapoints.

5. Iterate steps 3 and 4 for 1500 times.
  * In step 2, use 𝜇2 𝑎𝑛𝑑 𝜇4 values you computed in the previous iterations’ step 3. So at the end of these iterations, 
  you will have final values of means and information about which datapoint is assigned to which cluster. Print this result 
  in console.
