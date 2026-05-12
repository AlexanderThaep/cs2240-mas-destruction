**Group Name:** MAS Destruction (actually Jacobi Destruction)

**Group Members:** Alexander Thaep, Tanish Makadia

**Presentation Slides:** 

https://docs.google.com/presentation/d/1_hQpmnGfadxuZg-rJYMpV0KhI9LrQthv2ne7drXfGpY/edit?usp=sharing

<img width="500" height="750" alt="image" src="https://github.com/user-attachments/assets/4ce46baa-dc73-4b65-b09d-0d15c7adcfc4" />

**Papers**

[XPBD: Position-Based Simulation of Compliant Constrained Dynamics](https://dl.acm.org/doi/10.1145/2994258.2994272): We used the Lagrange multipliers from this paper for projecting collided voxels out of each other in a stable way. This paper specifically introduces a compliance term, without which our entire system goes kabloosky at small `dt` values. In particular, Eq. 18 from this paper shows up in our `project_collisions` function.

[A GPU-Based Multilevel Additive Schwarz Preconditioner for Cloth and
Deformable Body Simulation](https://dl.acm.org/doi/10.1145/3528223.3530085): Embarrassingly, in our tests we found that a simple Jacobi preconditioner, which just uses the diagonal of A, outperformed the MAS preconditioner introduced in this paper, so we never actually used MAS. We should probably be called "Jacobi Destruction" for that reason lol. However, we took a _lot_ of ideas from this paper into our voxel simulation. For one, the actual linear system described in section 3.1 is what we modeled our system after (neither of us were familiar with preconditioning before this, so this paper was kind of our introduction to it). Second, we repurposed the Morton codes that underpin the spatial-locality based optimizations of this paper's preconditioner to be used for short-listing nearby voxels that have the potential to collide (see `collision_candidates`).

[Animating Exploding Objects](https://dl.acm.org/doi/10.5555/351631.351688): This paper, while old and naive, was the original inspiration for this project altogether. We really didn't want to use FEM because we think voxels / hexahedral geometry is cool, and this was the first paper we found that had destruction with voxels. Some inspiration was taken from this paper when designing our `Voxels` class, but admittedly we deviate from this paper's grid based formulation a little. Originally, our voxel-voxel collisions were based on the centroid approach written here, but we came up with a better method that uses OBBs and the separating-axis theorem to handle collisions in a much more plausible way.

[Unified Particle Physics for Real-Time Applications](https://dl.acm.org/doi/10.1145/2601097.2601152): This paper is about particles, and while orthogonal to our voxel simulation intentions, offered a really neat way to obtain the orientation of voxels for use at collision time. Specifically, Eqs. 15 and 16 from this paper describe how you can decompose covariance matrices derived from current and rest positions (e.g. with SVD or polar decomposition) to obtain an orientation for our voxels. We use this almost verbatim in `voxel_rotations`, and ultimately this is how our OBBs actually become oriented.

[Fast Simulation of Mass-Spring Systems](https://dl.acm.org/doi/10.1145/2508363.2508406): All we took from this paper were the classical equations about Hooke's law and spring potential (see Eqs. 7,8,9). The actual method from this paper was not used, but it had been a while since either of us had dealt with spring-mass systems so this was nice just as a reference for the energy formulation.

[Stable but Responsive Cloth](https://dl.acm.org/doi/10.1145/566654.566624): Using the MAS paper (A GPU-Based Multilevel Additive Schwarz Preconditioner for Cloth and
Deformable Body Simulation) and solving for A and b using equations from the spring mass system paper (Fast Simulation of Mass-Spring Systems), we had this gross looking Hessian of energy that we didn't know how to write in PyTorch syntax. This paper also expresses the Hessian directly (see Eq. 3) and justifies why you can drop the higher-order terms and approximate it as a summation of rank-one matrices per edge of the form kdd^T (this comes from the text after Eq. 12).

[An O(log(n)) Parallel Connectivity Algorithm](https://www.sciencedirect.com/science/article/pii/0196677482900086?via=ihub): Rebuilding our `Voxels` data structure upon each and every fracture (there's thousands and thousands of them) is insanely expensive, and is part of why MAS didn't work as well as we'd hoped (i.e. rapid changes in topology slow things down). We found out that there's apparently a family of logarathmic time algorithms for computing the individual connected components of a graph structure based on the Shiloach-Vishkin (SV) connectivity algorithm (`undirected_components` relies on this).

**Other Resources**

[Voxel Models](https://github.com/ephtracy/voxel-model)

[Voxelizer](https://www.drububu.com/miscellaneous/voxelizer/index.html)

**Videos:**

[![video_1](https://i9.ytimg.com/vi/sYhCGDtH6VM/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALqBYoCDAgAEAEYfyAYKCYwDw==&rs=AOn4CLACQROPbEuxM0KaxNgB7nsmiE5l8Q)](https://youtube.com/shorts/sYhCGDtH6VM?feature=share)

[![video_2](https://i9.ytimg.com/vi/wt5kU33HZSk/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALqBYoCDAgAEAEYciBbKD8wDw==&rs=AOn4CLDJfY5z2O8c3NKGD2sKqQhcLjv29g)](https://youtube.com/shorts/wt5kU33HZSk?feature=share)

[![video_3](https://i9.ytimg.com/vi/67l1LIjbm1M/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALqBYoCDAgAEAEYfyAgKCkwDw==&rs=AOn4CLB33rIjZ5Vjn8Aeul7BCER3DixToA)](https://youtube.com/shorts/67l1LIjbm1M?feature=share)

[![video_4](https://i9.ytimg.com/vi/KiJJFahelKk/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALqBYoCDAgAEAEYfyAYKCYwDw==&rs=AOn4CLDK3uxnuye5xPMqECm1PIEdJ5r8WQ)](https://youtube.com/shorts/KiJJFahelKk?feature=share)

[![video_5](https://i9.ytimg.com/vi/phJEUnF8o3c/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4AbYIgALeCIoCDAgAEAEYTyBbKGUwDw==&rs=AOn4CLBqVjwqCe3NBNIZ-GhS73EoGpbQfw)](https://youtube.com/shorts/phJEUnF8o3c?feature=share)

[![video_6](https://i9.ytimg.com/vi/Erwu2HWaMtA/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALqBYoCDAgAEAEYQiBbKHIwDw==&rs=AOn4CLDHB7Fauv8wEi15G5wqb6osc8Dteg)](https://youtube.com/shorts/Erwu2HWaMtA?feature=share)

[![video_7](https://i9.ytimg.com/vi/nXORahhBeXM/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALqBYoCDAgAEAEYciBqKBEwDw==&rs=AOn4CLDq42pYs59L8bZ4xdWPGUVBOIdN2Q)](https://youtube.com/shorts/nXORahhBeXM?feature=share)

[![video_8](https://i9.ytimg.com/vi/MCezF08tC7Q/mq2.jpg?sqp=CPiXitAG-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALqBYoCDAgAEAEYZSBlKGUwDw==&rs=AOn4CLCoujno5dJizKaRyJ1pAC7MXniRBA)](https://youtube.com/shorts/MCezF08tC7Q?feature=share)

[![video_9](https://github.com/user-attachments/assets/6abe209c-0b19-4841-817c-bc17cc69dc6f)](https://www.youtube.com/watch?v=fJmOybk7moU)

[![video_10](https://github.com/user-attachments/assets/50d6f1e7-5145-4dc9-a7c4-ad386452e5bf)](https://www.youtube.com/watch?v=xJoDtrYiGpU)

[![video_11](https://github.com/user-attachments/assets/dba4244f-f1b3-4a2b-9023-bfe4ac701cb2)](https://www.youtube.com/watch?v=gTvdI11A6lo)

**Don't forget to do set up a virtual environment and run `python setup.py` first (or pip install requirements manually)**
