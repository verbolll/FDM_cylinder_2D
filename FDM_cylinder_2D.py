import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def solve_cfd():

    history_cd = []
    history_cl = []
    timestamps = []

    Lx, Ly = 8.0, 2.0         # 计算域长宽
    Nx, Ny = 400, 100         # 网格分辨率 (越高越精确，但越慢)
    dx, dy = Lx / Nx, Ly / Ny # 网格间距
    
    rho = 1.0                 # 密度
    nu = 0.01                 # 运动粘度
    inlet_u = 2.0             # 入口速度
    
    dt = 0.001                # 时间步长 (必须很小以保证稳定, CFL条件)
    nt = 20000                 # 总时间步数
    
    # 圆柱障碍物设置
    cylinder_cx, cylinder_cy = 2.0, 1.0 # 圆心位置
    cylinder_r = 0.25                   # 半径


    # 网格与初始化

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # 初始化场变量
    u = np.zeros((Ny, Nx))    # x方向速度
    v = np.zeros((Ny, Nx))    # y方向速度
    p = np.zeros((Ny, Nx))    # 压力
    b = np.zeros((Ny, Nx))    # 泊松方程源项

    v[40:60, 40:60] = 0.1

    # 圆柱: 圆柱内部为 True
    mask = (X - cylinder_cx)**2 + (Y - cylinder_cy)**2 < cylinder_r**2


    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 4))

    # Colorbar
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
        
        # 施加边界条件
        # 1. Inlet: 左侧固定速度
        u[:, 0] = inlet_u
        v[:, 0] = 0
        
        # 2. Outlet: 右侧零梯度流体流出
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        
        # 3. Walls:Symmetry
        v[0, :] = 0
        v[-1, :] = 0

        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        
        # 4. cylinder
        u[mask] = 0
        v[mask] = 0
      
        uc = u[1:-1, 1:-1]
        vc = v[1:-1, 1:-1]
        
        uL = u[1:-1, 0:-2]; uR = u[1:-1, 2:]
        uD = u[0:-2, 1:-1]; uU = u[2:, 1:-1]
        vL = v[1:-1, 0:-2]; vR = v[1:-1, 2:]
        vD = v[0:-2, 1:-1]; vU = v[2:, 1:-1]

        # 离散化 N-S 方程
        # du/dt + u*du/dx + v*du/dy = nu*(d2u/dx2 + d2u/dy2)
        
        # 对流项 - 直接中心差分
        du_dx = (uR - uL) / (2*dx)
        du_dy = (uU - uD) / (2*dy)
        dv_dx = (vR - vL) / (2*dx)
        dv_dy = (vU - vD) / (2*dy)
        
        # 扩散项 - 拉普拉斯算子
        lap_u = (uR - 2*uc + uL)/dx**2 + (uU - 2*uc + uD)/dy**2
        lap_v = (vR - 2*vc + vL)/dx**2 + (vU - 2*vc + vD)/dy**2

        # 临时速度 u_star
        u_star = uc + dt * (- (uc * du_dx + vc * du_dy) + nu * lap_u)
        v_star = vc + dt * (- (uc * dv_dx + vc * dv_dy) + nu * lap_v)

        u[1:-1, 1:-1] = u_star
        v[1:-1, 1:-1] = v_star
        
        # cylinder
        u[mask] = 0
        v[mask] = 0

        # 求解压力泊松方程
        # div(u*) = laplacian(p) * dt / rho
        
        # 1. 计算散度
        div_u = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2*dx) + \
                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2*dy)
        
        b[1:-1, 1:-1] = (rho / dt) * div_u

        # 2. 雅可比迭代求p
        pressure_iter = 50 
        for _ in range(pressure_iter):
            p[1:-1, 1:-1] = ( (p[1:-1, 2:] + p[1:-1, 0:-2]) * dy**2 + 
                              (p[2:, 1:-1] + p[0:-2, 1:-1]) * dx**2 - 
                              b[1:-1, 1:-1] * dx**2 * dy**2 ) / (2 * (dx**2 + dy**2))
            
            p[:, -1] = 0         # 出口压力为0 (参考压力)
            p[:, 0]  = p[:, 1]   # 入口 dp/dx = 0
            p[0, :]  = p[1, :]   # 下壁面 dp/dy = 0
            p[-1, :] = p[-2, :]  # 上壁面 dp/dy = 0

        # 速度修正
        # u_new = u_star - (dt/rho) * grad(p)
        
        dp_dx = (p[1:-1, 2:] - p[1:-1, 0:-2]) / (2*dx)
        dp_dy = (p[2:, 1:-1] - p[0:-2, 1:-1]) / (2*dy)
        
        u[1:-1, 1:-1] -= (dt / rho) * dp_dx
        v[1:-1, 1:-1] -= (dt / rho) * dp_dy
        
        # 计算升阻力
        # F = sum( rho * (u_current - 0) / dt * dA )
        
        # 只提取圆柱内部的速度
        u_force = u[mask]
        v_force = v[mask]
        
        # 总受力 (F = ma)
        # u_force/dt加速度，rho*dx*dy质量
        fx = np.sum(rho * u_force / dt * dx * dy)
        fy = np.sum(rho * v_force / dt * dx * dy)
        
        # Cd = Fx / (0.5 * rho * U^2 * D)
        # Cl = Fy / (0.5 * rho * U^2 * D)
        D = 2 * cylinder_r
        U_inf = inlet_u
        den = 0.5 * rho * U_inf**2 * D
        
        cd = fx / den
        cl = fy / den
        
        history_cd.append(cd)
        history_cl.append(cl)
        timestamps.append(n * dt)
        
        u[mask] = 0
        v[mask] = 0

        if n % 50 == 0:
            ax.clear()
            
            # 计算涡量 (Vorticity) = dv/dx - du/dy
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

    # 升阻力曲线
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
