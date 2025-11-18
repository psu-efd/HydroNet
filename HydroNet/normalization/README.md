# Data Normalization for DeepONet, PINN, and Physics-informed DeepONet models for 2D computational hydraulic models

This is the normalization module for the HydroNet project. 

For DeepONet, PINN, and Physics-informed DeepONet models for 2D computational hydraulic models, we need to prepare and normalize data. The data include coordinates ($x$, $y$, $t$), solution variables ($h$, $u$, $v$), and other variables such as bed elevation $z_b$, Manning's $n$, bed slope ($s_x$, $s_y$). We need to consider what data normalization methods (min-max, z-score, etc) to use, should we use one method for all variables or we can use different method for different variables, and what factors should we consider and how they affect the training and model performance. Here is a detailed discussion on this topic.

For a DeepONet on 2D shallow-water–type data, I’d _definitely_ not treat “all variables the same.” Different groups of variables should be normalized differently, and that choice _does_ affect stability, speed of convergence, and ultimately accuracy/generalization.

Let's break it down in a way that’s practical for our use case.

---
## 1. Separate variables into _roles_

We have several _types_ of variables:

- **Input coordinates:**
    
    - ($x$, $y$) (space)
        
    - ($t$) (time)
        
- **Solution variables:**
    
    - ($h$) (water depth or stage)
        
    - ($u$, $v$) (velocity components)
        
- **Static / parametric fields:**
    
    - ($z_b$) (bed elevation)
        
    - ($n$) (Manning’s roughness)
        
    - ($s_x$, $s_y$) (bed slopes or source terms)
        

These don't play the same role and don't have the same scale or units, so it's usually _good_ to normalize them differently (but consistently within each group).

---

## 2. General principles for normalization

Some rules of thumb:

1. **Avoid extremely different scales across inputs to the _same_ network.**
    
    - Orders of magnitude differences (e.g., ($x \sim 10^3$ m), ($h \sim 1$ m), ($n \sim 0.03$)) can lead to poorly conditioned optimization, exploding/vanishing gradients.
        
2. **Keep outputs roughly ($\mathcal{O}(1)$).**
    
    - DeepONet branch/trunk nets learn more smoothly when target outputs are in something like ([-2,2]) or ([-1,1]).
        
3. **Preserve “important” structure when needed.**
    
    - E.g., sign, relative differences, boundaries of the domain.
        
    - For spatial coordinates, it’s often useful to map the domain to a fixed reference like ([-1,1]$^2$).
        
4. **Think about deployment:**
    
    - Whatever normalization we choose must be _easily invertible_ and applied consistently in inference.
        

---

## 3. Recommended choices by variable type

### (A) Coordinates: ($x$, $y$, $t$)

**Goal:** Make the domain roughly ([-1,1]$^d$) or ([0,1]$^d$) for stability and translation across cases.

**Good options:**

- **Option A1: Linear scaling to ([-1,1])**  
    For each dimension separately:  
    $$
    \tilde{x} = 2\frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1  
    $$
    Similarly for ($y$, $t$).  
    If you’d like the DeepONet to generalize across _multiple domains_ with different sizes:
    
    - Consider scaling coordinates by a characteristic length/time scale (e.g., ($x^* = x / L$), ($t^* = t / T$)) that is _shared_ across cases.
        
- **Option A2: Center and scale by a characteristic scale (z-score-like, but physics-based)**  
    $$
    \tilde{x} = \frac{x - x_0}{L}, \quad \tilde{y} = \frac{y - y_0}{L}, \quad \tilde{t} = \frac{t - t_0}{T}  
    $$
    where ($x_0$, $y_0$) is domain center, ($L$) a characteristic length, ($t_0$) an initial time, ($T$) a characteristic time (e.g., duration of hydrograph).
    

**I’d recommend:**

- **Use the same scaling rule for ($x$, $y$) across all cases** (e.g., all domains mapped to ([-1,1]$^2$) or “centered and divided by L”).
    
- For ($t$), often ([0,1]) or ([-1,1]) mapping over the simulation duration is fine.
    

**Why it matters:**

- Helps the trunk net learn smooth basis functions in a standardized domain.
    
- DeepONet papers often implicitly assume scaled coordinates; if not, training can be harder.
    

---

### (B) Outputs: ($h$, $u$, $v$)

These are what DeepONet will predict. It’s typically best to give the network targets with unit-ish variance and modest range.

**Common choice:**

- **Z-score normalization per variable** (across the whole training set):  
    $$
    \tilde{h} = \frac{h - \mu_h}{\sigma_h}, \quad \tilde{u} = \frac{u - \mu_u}{\sigma_u}, \quad \tilde{v} = \frac{v - \mu_v}{\sigma_v}  
    $$
    where (\mu) and (\sigma) are computed over _all_ training samples (space, time, cases).
    

**Alternative:**

- **Min–max to ([-1,1])** for each variable:  
    $$
    \tilde{h} = 2\frac{h - h_{\min}}{h_{\max} - h_{\min}} - 1  
    $$
    etc. This makes them compact and well-bounded but can be sensitive to outliers (e.g., very high velocities in a local jet).
    

**What I’d do for shallow water:**

- If distributions are reasonably “nice” and no crazy outliers: **z-score** works very well.
    
- If you expect big outliers or extremely skewed distributions: consider
    
    - **Clipping extreme values** before computing ($\mu$, $\sigma$), or
        
    - Using a **robust scaling** (e.g., median and IQR).
        

**How it affects training:**

- Well-scaled outputs make the loss landscape smoother and gradients more balanced across ($h$, $u$, $v$).
    
- If one variable is much larger (e.g., ($u$) dominates scale), it will dominate the loss unless you compensate with loss weights.
    

---

### (C) Static fields: ($z_b$, $n$, $s_x$, $s_y$)

These act like _input parameter fields_ to the branch net. They can have very different units and ranges.

I’d treat them as _features_, each with its own normalization. Don’t force them to share one min–max/z-score.

#### Bed elevation ($z_b$)

Typical range can be tens of meters, plus there can be an arbitrary datum.

Options:

- **Subtract a reference and scale:**
    
    - E.g., define ($z_b' = z_b - z_{\text{ref}}$), where ($z_{\text{ref}}$) could be:
        
        - domain-mean elevation, or
            
        - a global reference (e.g., min bed over all training cases).
            
    - Then either:
        
        - z-score ($z_b'$) (global mean/std)
            
        - or min–max to ([-1,1]).
            

This makes the network focus on _relative_ bed variations rather than absolute datum, which is usually what matters hydraulically.

#### Manning’s (n)

Range is small (e.g., 0.01–0.10). A simple approach:

- **Either:**
    
    - Min–max to ([0,1]) or ([-1,1]), or
        
    - z-score (but watch that the variance might be tiny).
        

Because ($n$) often takes only a few discrete values (land use classes), you could also treat it as categorical in some advanced setups, but for now: simple continuous scaling is fine.

#### Slopes ($s_x$, $s_y$)

These can be small (e.g., ($10^{-4}–10^{-2}$)), possibly positive/negative.

- Good approach: **z-score per component**, or scale by a reference slope ($S_0$) (e.g., ($10^{-3}$)):  
    $$
    \tilde{s}_x = \frac{s_x}{S_0},\quad \tilde{s}_y = \frac{s_y}{S_0}  
    $$
    
- If you do z-score, make sure the values are not too close to numerical noise.
    

---

## 4. Use _different_ normalization for different groups

So to our direct question:

> Should we use one method for all variables or different methods?

I would **absolutely use different normalizations for different groups**:

- **Coordinates**: map to ([-1,1]$^2$) or ([0,1]$^2$) using linear scaling / physics-based nondimensionalization.
    
- **Outputs** (($h$, $u$, $v$)): z-score or min–max per variable.
    
- **Static fields** (($z_b$, $n$, $s_x$, $s_y$)): individual scaling, likely z-score or min–max per field.
    

What you _should_ keep consistent:

- Within a group across all training cases – e.g., same normalization formula for ($h$) across all simulations.
    
- Between train/validation/test – store the stats and reuse them.
    

---

## 5. Practical factors to consider

Here are the key knobs and their effects:

### (1) Scale and units across variables

- If a variable has much larger magnitude than others and we don’t normalize properly, the model will “focus” on that variable in both input and output.
    
- Proper scaling avoids the need to hand-tune loss weights as aggressively.
    

### (2) Distribution shape (skew, heavy tails)

- **Z-score** works well if the distribution is approximately symmetric (or at least not crazy skewed).
    
- **Min–max** can be fragile if we have rare extremes; those extremes will map to ($\pm 1$) and marginal values get compressed.
    
- If we have strong skew (e.g., many nearly-dry cells with ($h \approx 0$), and a few deep zones), consider:
    
    - Transform: e.g., log or sqrt on ($h$) (but then handle dry cells carefully with ($\epsilon$)).
        
    - Or treat dry cells specially.
        

### (3) Multi-case training

We’re probably training DeepONet across many SRH-2D cases:

- Compute normalization statistics **over all cases** (global stats) to ensure consistent scaling.
    
- If some cases are very different (e.g., mountainous vs coastal), you may need:
    
    - Different sub-models, or
        
    - Normalization conditioning (e.g., per-case scaling passed as additional input).
        

### (4) Nondimensionalization vs. pure normalization

It may be advantageous to:

- Nondimensionalize variables using physical scales:
    
    - ($h^* = h / H_0$)
        
    - ($u^* = u / U_0$)
        
    - ($t^* = t / T_0$)
        
- Then maybe **mild** additional normalization (e.g., center around zero).
    

This:

- Makes the learning problem closer to learning dimensionless relationships,
    
- Can improve cross-scenario generalization (e.g., different rivers, scales).
    

### (5) Effect on training stability

Good normalization tends to:

- Allow larger learning rates without divergence.
    
- Reduce gradient anisotropy, so you converge in fewer epochs.
    
- Reduce sensitivity to initialization and optimizer hyperparams.
    

Bad/uneven normalization can manifest as:

- Loss stagnant in first epochs,
    
- One output dimension learned well, others lagging badly,
    
- Numerical issues (exploding gradients).
    

---

## 6. Suggested concrete recipe for you (first pass)


1. **Coordinates:**
    
    - For each case, map ($x$, $y$) to ([-1,1]$^2$) using global ($x_{\min}$, $x_{\max}$) and ($y_{\min}$, $y_{\max}$) across all cases.
        
    - Map ($t$) to ([0,1]) over the simulation duration for each case, or use a global time scale if durations are similar.
        
2. **Static fields:**
    
    - ($z_b$): subtract global mean elevation over all cells and cases, divide by global std → z-score.
        
    - ($n$): z-score or min–max to ([0,1]) over all grid cells and cases.
        
    - ($s_x$, $s_y$): z-score over all cells and cases.
        
3. **Outputs:**
    
    - For ($h$, $u$, $v$):
        
        - Compute global ($\mu$) and ($\sigma$) across all training samples.
            
        - Use z-score for each variable:  
            $$
            \tilde{h}, \tilde{u}, \tilde{v} = \frac{\cdot - \mu}{\sigma}. 
            $$
            
            
    - During inference, remember to invert:  
        $$
        h = \tilde{h},\sigma_h + \mu_h,\ \text{etc.}  
        $$
        
4. **Record everything:**  
    Save your normalization parameters in YAML alongside the trained model so your deployment pipeline can reproduce them.
    

