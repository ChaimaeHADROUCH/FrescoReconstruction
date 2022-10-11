

## Objective :
- Reconstruction of digitized frescoes.

## Problematic :
- Template matching with no view change.

## Limitations :
- Large number of fragments
- Irregular shape of fragments
- Uniqueness and non-overlapping constraints



![image](https://drive.google.com/uc?export=view&id=1I2cw9mmoZV90d6DcnYEO58wsJYMRKb61)

# Process

## Detection of keypoints

![image](https://drive.google.com/uc?export=view&id=1KyiaaADD5S9ESpmB6NY0mfCNtnlrGIGU)

## Matching

![image](https://drive.google.com/uc?export=view&id=10o9p6pIB38nxfv7WWuvmMVKCj1r-Gtsa)

## Affine Transformation

![image](https://drive.google.com/uc?export=view&id=1LwTt8Z4QuIsyMGoDf_wzN41WCGXPjegI)

# Reconstruction

![image](https://drive.google.com/uc?export=view&id=1JPpIxcMLkTvK-DPYhtyZboI9z6IQRKLg)

# Visualization of results

![image](https://drive.google.com/uc?export=view&id=12W4aDQ--7BEYbDLNI2eaaoOt8yEhQNqh)



![image](https://drive.google.com/uc?export=view&id=14FmsuA_AahepGDWaBW-Htt1tupUunQh0)

# Perspectives and conclusion

## SuperGlue+ SuperPoint
- The method doesn’t need learning.

- Parameters can be tuned to improve performance

- SuperGlueplaces the fragments in a precise way on the fresco, but it places less.

## D2Net

- Creation of a database of ground truth to do the Learning.

- To do this, we must calculate the keypoints using the classic methods (+ random keypoints with a rotation of the fragments to increase the database)

- D2NET places many fragments but in a less precise way.
