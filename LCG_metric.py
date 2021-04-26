import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.spatial
import time


def LCG_metric(obs,method='analytic',tavg=None,scan_avg=False,dummy_circ=True,plot_solution=False,niter=200,specify_x0=None):
    #input:
    #obs: eht-imaging obsdata uvfits file with the coverage
    #method: 'analytic' for Delaunay triangulation-based algorithm, 'numeric' for basin hopping heuristic
    #tavg: averaging time for the coverage
    #scan_avg: whether we average data by scan
    #dummy_circ: whether we should add dummy points on the coverage circumference to avoid 
    #finding a gap extending outside of the covarage (recommended)
    #plot_solution: should we make a plot visualizing the found gap
    #niter: number of annealing basin hopping hops for the numerical solver
    #specify_x0: put initial condition for the numerical solver by hand (r,phi)

    #output:
    #metric: gap metric = 1 - largest_gap_diameter/max_uvdist
    
    #Maciek Wielgus, Daniel Palumbo, Linus Hamilton 2020
    
    start0 = time.time()
    if tavg: obs = obs.avg_coherent(inttime=tavg,scan_avg=scan_avg)
    
    data= obs.unpack(['time','u','v','uvdist','t1','t2','rramp','llamp','rrsigma','llsigma'])
    uvec = [x[1]/1e9 for x in data]
    vvec = [x[2]/1e9 for x in data]
    uvecconj = [-x[1]/1e9 for x in data]
    vvecconj = [-x[2]/1e9 for x in data]
    uvec = uvec+uvecconj
    vvec = vvec+vvecconj
    maxuv=max(np.array([x[3]/1e9 for x in data]))

    #add dummy points on the circumference (recommended)
    if dummy_circ:
        Ndummy=256
        tdummy = np.linspace(0,2*np.pi,Ndummy)
        ucirc = maxuv*np.sin(tdummy)
        vcirc = maxuv*np.cos(tdummy)
        uvecwork=np.array(uvec+list(ucirc))
        vvecwork=np.array(vvec+list(vcirc))
    else:
        uvecwork=np.array(uvec)
        vvecwork=np.array(vvec)

    
    start = time.time()
    if method=='analytic':
        center, rgap = largestEmptyCircle(uvecwork, vvecwork)
        r0 = np.sqrt(center[0]**2+center[1]**2)
        phi0 = np.angle(center[1]+center[0]*1j)
    elif method=='numeric':
        if specify_x0: x0 = specify_x0
        else: x0 = (maxuv/2.0,np.pi/2.)
        #set up the minimizer and solve for the most remote point
        bnds=((0.05,maxuv),(0,np.pi))
        minimizer_kwargs = { "method": "L-BFGS-B","bounds":bnds,"tol":1.e-4}
        def distance_from_coverage(x):
            return -np.min(((uvecwork-x[0]*np.sin(x[1]))**2 + (vvecwork-x[0]*np.cos(x[1]))**2))
        sol = so.basinhopping(distance_from_coverage, x0,T=maxuv/10.0,niter=niter,minimizer_kwargs=minimizer_kwargs)
        rgap = np.sqrt(-sol.fun)
        r0 = sol.x[0]
        phi0=sol.x[1]
    
    metric = 1.-2.0*rgap/maxuv
    stop = time.time()
    
    #output info
    print('solver time:', stop-start )
    print('total time:', stop-start0 )
    print('largest diameter (Gigalambda):', 2.0*rgap)
    print('LCG metric:', metric)
    
    #plot visualization
    if plot_solution:
        plt.figure(figsize=(10,10))
        plt.plot(uvec,vvec,'o',ms=2,color='b')
        #rgap = np.sqrt(-sol.fun)
        
        t = np.linspace(0,2*np.pi,200)
        plt.plot(rgap*np.cos(t)-r0*np.sin(phi0),rgap*np.sin(t)-r0*np.cos(phi0),'r')
        plt.plot(rgap*np.cos(t)+r0*np.sin(phi0),rgap*np.sin(t)+r0*np.cos(phi0),'r')
        plt.plot(maxuv*np.cos(t),maxuv*np.sin(t),'b')
        plt.plot(r0*np.sin(phi0),r0*np.cos(phi0),'+k',ms=15,mew=3)
        plt.plot(-r0*np.sin(phi0),-r0*np.cos(phi0),'+k',ms=15,mew=3)
        plt.xlabel(r'u')
        plt.ylabel(r'v')
        plt.show()
    
    return metric


def largestEmptyCircle(xs, ys):
    """ Finds the largest empty circle, among the points (x,y),
    which cannot freely slide without hitting a point. """
    points = np.stack([xs, ys], axis=1)
    triangulation = scipy.spatial.Delaunay(points)
    triangleIndices = triangulation.simplices
    # `triangleIndices` is an array with shape (?,3)
    # whose rows are (i,j,k) indices of the vertices
    # (points[i], points[j], points[k]) in each Delaunay triangle.
    triangles = points[triangleIndices] # Wow, numpy indexing is cool
    # Remove strictly obtuse triangles, since they can always "slide"
    nonobtuseTriangles = removeObtuse(triangles)
    # Each Delaunay triangle begets a circumcircle containing no other points.
    # Now we just find the biggest one.
    circleRadii = circumradii(nonobtuseTriangles)
    biggestCircleIndex = np.argmax(circleRadii)
    return (circumcenter(nonobtuseTriangles[biggestCircleIndex]),
            circleRadii[biggestCircleIndex])

def circumcenter(triangle):
    """ `triangle`: an array of shape (3,2)
    returns an array of shape (2,): the incenter of the triangle. """
    # get side lengths
    A,B,C = triangle
    a,b,c = map(np.linalg.norm, (B-C, C-A, A-B))
    # use barycentric coordinate formula
    aw,bw,cw = (a**2*(b**2+c**2-a**2),
                b**2*(c**2+a**2-b**2),
                c**2*(a**2+b**2-c**2))
    return (aw*A + bw*B + cw*C)/(aw+bw+cw)

def removeObtuse(triangles):
    """ `triangles`: an array of shape (?,3,2)
    returns an array of shape (something,3,2) removing obtuse triangles. """
    # get side lengths
    A,B,C = triangles[:,0,:], triangles[:,1,:], triangles[:,2,:]
    a,b,c = [np.linalg.norm(vecs, axis=1) for vecs in (A-B,B-C,C-A)]
    # Give a little buffer, so we don't accidentally remove right triangles
    epsilon = 1e-6
    # A triangle is obtuse if (biggest side)^2 is bigger than the sum of
    # the squares of the other two sides
    notObtuse = (a**2+b**2-c**2 > -epsilon * (a+b+c)**2) &\
                (b**2+c**2-a**2 > -epsilon * (a+b+c)**2) &\
                (c**2+a**2-b**2 > -epsilon * (a+b+c)**2)
    return triangles[notObtuse]
    
def circumradii(triangles):
    """ `triangles`: an array of shape (?,3,2)
    returns an array of shape (?,) with the circumradii of the triangles. """
    # get side lengths
    A,B,C = triangles[:,0,:], triangles[:,1,:], triangles[:,2,:]
    a,b,c = [np.linalg.norm(vecs, axis=1) for vecs in (A-B,B-C,C-A)]
    # use formula
    return a*b*c/((a+b+c)*(a+b-c)*(b+c-a)*(c+a-b))**0.5