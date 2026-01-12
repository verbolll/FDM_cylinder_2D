import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
圆柱强迫运动
"""

def solve_cfd():

    history_cd = []
    history_cl = []
    timestamps = []

    Lx, Ly = 8.0, 2.0         
    Nx, Ny = 400, 100         
    dx, dy = Lx / Nx, Ly / Ny 
    
    rho = 1.0                 
    nu = 0.01                 
    inlet_u = 2.0             
    
    dt = 0.001                
    nt = 50000                
    
    cylinder_cx, cylinder_cy = 2.0, 1.0 
    cylinder_r = 0.25                   

    # 新增：振动参数
    amp = 0.5       # 振幅 A
    freq = 0.2      # 频率 f
    # 圆柱运动方程: y(t) = 1.0 + A * sin(2 * pi * f * t)
    # 圆柱速度方程: v(t) = A * 2 * pi * f * cos(2 * pi * f * t)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    p = np.zeros((Ny, Nx))
    b = np.zeros((Ny, Nx))

    v[40:60, 40:60] = 0.1

    mask = (X - cylinder_cx)**2 + (Y - cylinder_cy)**2 < cylinder_r**2

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 4))

    ax = fig.add_axes([0.08, 0.1, 0.8, 0.8])
    cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
    
    levels = np.linspace(-10, 10, 21)
    dummy_cont = ax.contourf(X, Y, np.zeros_like(u), 
                             levels=levels, 
                             cmap='coolwarm', 
                             extend='both')
    cb = plt.colorbar(dummy_cont, cax=cax)
    cb.set_label(r'Vorticity ($\omega$)')

    print(f"网格大小: {Nx}x{Ny}, 雷诺数 Re ≈ {inlet_u * 2*cylinder_r / nu}")

    for n in tqdm(range(nt), desc="Computing"): 


        # 新增：计算当前时刻圆柱的位移和速度
        curr_time = n * dt
        
        # 圆心位置
        cylinder_cy = 1.0 + amp * np.sin(2 * np.pi * freq * curr_time)
        
        # 圆柱速度
        v_cyl = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * curr_time)
        
        mask = (X - cylinder_cx)**2 + (Y - cylinder_cy)**2 < cylinder_r**2

        u[:, 0] = inlet_u
        v[:, 0] = 0

        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]

        v[0, :] = 0
        v[-1, :] = 0

        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]

        u[mask] = 0    
        v[mask] = v_cyl
      
        uc = u[1:-1, 1:-1]
        vc = v[1:-1, 1:-1]
        
        uL = u[1:-1, 0:-2]; uR = u[1:-1, 2:]
        uD = u[0:-2, 1:-1]; uU = u[2:, 1:-1]
        vL = v[1:-1, 0:-2]; vR = v[1:-1, 2:]
        vD = v[0:-2, 1:-1]; vU = v[2:, 1:-1]
        
        du_dx = (uR - uL) / (2*dx)
        du_dy = (uU - uD) / (2*dy)
        dv_dx = (vR - vL) / (2*dx)
        dv_dy = (vU - vD) / (2*dy)
        
        lap_u = (uR - 2*uc + uL)/dx**2 + (uU - 2*uc + uD)/dy**2
        lap_v = (vR - 2*vc + vL)/dx**2 + (vU - 2*vc + vD)/dy**2

        u_star = uc + dt * (- (uc * du_dx + vc * du_dy) + nu * lap_u)
        v_star = vc + dt * (- (uc * dv_dx + vc * dv_dy) + nu * lap_v)

        u[1:-1, 1:-1] = u_star
        v[1:-1, 1:-1] = v_star
        
        u[mask] = 0
        v[mask] = v_cyl  # 当前圆柱速度
        
        div_u = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2*dx) + \
                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2*dy)
        
        b[1:-1, 1:-1] = (rho / dt) * div_u

        pressure_iter = 50 
        for _ in range(pressure_iter):
            p[1:-1, 1:-1] = ( (p[1:-1, 2:] + p[1:-1, 0:-2]) * dy**2 + 
                              (p[2:, 1:-1] + p[0:-2, 1:-1]) * dx**2 - 
                              b[1:-1, 1:-1] * dx**2 * dy**2 ) / (2 * (dx**2 + dy**2))
            
            p[:, -1] = 0       
            p[:, 0]  = p[:, 1] 
            p[0, :]  = p[1, :] 
            p[-1, :] = p[-2, :]
        
        dp_dx = (p[1:-1, 2:] - p[1:-1, 0:-2]) / (2*dx)
        dp_dy = (p[2:, 1:-1] - p[0:-2, 1:-1]) / (2*dy)
        
        u[1:-1, 1:-1] -= (dt / rho) * dp_dx
        v[1:-1, 1:-1] -= (dt / rho) * dp_dy

        u_force = u[mask]

        v_fluid = v[mask]
        
        fx = np.sum(rho * u_force / dt * dx * dy)
        fy = np.sum(rho * (v_fluid - v_cyl) / dt * dx * dy)

        D = 2 * cylinder_r
        U_inf = inlet_u
        den = 0.5 * rho * U_inf**2 * D
        
        cd = fx / den
        cl = fy / den
        
        history_cd.append(cd)
        history_cl.append(cl)
        timestamps.append(n * dt)
        
        u[mask] = 0
        v[mask] = v_cyl

        if n % 50 == 0:
            ax.clear()
            
            vorticity = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2*dx) - \
                        (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2*dy)
            
            u_inner = u[1:-1, 1:-1]
            v_inner = v[1:-1, 1:-1]
            
            speed = np.sqrt(u_inner**2 + v_inner**2)

            levels = np.linspace(-10, 10, 100)
            cont = ax.contourf(X[1:-1, 1:-1], Y[1:-1, 1:-1], vorticity, 
                   levels=levels,
                   cmap='coolwarm',
                   extend='both')
            
            circle = plt.Circle((cylinder_cx, cylinder_cy), cylinder_r, color='black')
            ax.add_patch(circle)
            
            ax.set_title(f"Step {n}")
            ax.set_aspect('equal')
            plt.pause(0.001)
            plt.savefig(rf'./pic/{str(n)}.png')

    plt.ioff()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, history_cd, label='$C_D$ (Drag)', color='red')
    plt.plot(timestamps, history_cl, label='$C_L$ (Lift)', color='blue', alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    solve_cfd()
