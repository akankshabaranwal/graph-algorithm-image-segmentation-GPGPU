# Felzenszwalb using Boruvka

Idea: maybe iteratively grow k to first encourage joining of similar components?

```jsx
**Initialise** all components C_i <- V_i
**until** no components are joined anymore  
	**for** each component C_j
		(u,v) <- C_j.cheapest_edge

	if w(u,v) <= min(Int(C_u) + k/|C_u|, Int(C_v) + k/|C_v|)
		merge C_u and C_v
```

**Note:** output will inherently not resemble original algorithm completely because the threshold function that depends the component size to decide whether to join two components. This makes it a little bit different from the normal MST that gives of course the exact same output.

### Options

Outputs for sigma=0.7, k=500, min=50. 1 run

  0.  **original** - 7.27 seconds

![Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled.png](Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled.png)

1. **get cheapest edges as candidates, then decide whether to join these using predicate. Don't update component size used for predicate for different joins in same iteration -** 14.32 seconds
    - seriously undersegments for same as original, but seems more similar to a real parallel version without much synchronisation

        ![Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled%201.png](Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled%201.png)

2. **get cheapest edges as candidate, then decide whether to join using predicate. Update component size used for predicate for different joins in same iteration.** - 13.07 seconds
    - Segmentation looks a lot more similar to original

    ![Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled%202.png](Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled%202.png)

    1 & 2 resemble original algorithm more and also faster. 1 seems more similar to parallel version but seriously undersegments

3. **get cheapest edge that satisfies predicate as candidate, then join all these - 14.64 seconds**
    - Resembles original algorithm less than option 1 and 2
    - Also undersegments (although less than option 1) for same k as original

    ![Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled%203.png](Felzenszwalb%20using%20Boruvka%2046793af8479747819a667ca606cf40e6/Untitled%203.png)