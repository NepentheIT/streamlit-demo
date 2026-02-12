import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import platform

# ================= 1. 解决中文乱码与负号显示问题 =================
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system_name == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 页面配置 =================
st.set_page_config(page_title="STP 多模态融合实验室", layout="wide", page_icon="🧬")

# 自定义CSS
st.markdown("""
<style>
    .step-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #4e8cff;
    }
</style>
""", unsafe_allow_html=True)


# ================= 3. 核心数学逻辑函数 (只定义一次!) =================

def get_lcm(a, b):
    """计算最小公倍数"""
    if a == 0 or b == 0: return 0
    return abs(a * b) // math.gcd(a, b)


def stp_product_stepwise(A, B):
    """返回STP乘法的每一步状态"""
    m, n = A.shape
    p, q = B.shape
    L = get_lcm(n, p)
    alpha = L // n
    beta = L // p

    # 步骤1: 扩张
    id_alpha = np.eye(alpha, dtype=int)
    id_beta = np.eye(beta, dtype=int)
    A_kron = np.kron(A, id_alpha)
    B_kron = np.kron(B, id_beta)

    # 步骤2: 乘法
    Result = np.dot(A_kron, B_kron)

    return {
        "LCM": L, "alpha": alpha, "beta": beta,
        "A_kron": A_kron, "B_kron": B_kron, "Result": Result
    }


def stp_addition_stepwise(V1, V2):
    """返回STP加法（特征融合）的每一步状态"""
    m = V1.shape[0]
    p = V2.shape[0]
    L = get_lcm(m, p)
    alpha = L // m
    beta = L // p

    # 步骤1: 扩张 (使用全1向量做 Kronecker 积)
    ones_alpha = np.ones((alpha, 1), dtype=int)
    ones_beta = np.ones((beta, 1), dtype=int)

    V1_kron = np.kron(V1, ones_alpha)
    V2_kron = np.kron(V2, ones_beta)

    # 步骤2: 加法
    Result = V1_kron + V2_kron

    return {
        "LCM": L, "alpha": alpha, "beta": beta,
        "V1_kron": V1_kron, "V2_kron": V2_kron, "Result": Result
    }


# ================= 4. 可视化辅助函数 =================

def draw_heatmap(data, title, cmap="Blues", annot=True):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(data, annot=annot, fmt='d', cmap=cmap, cbar=False,
                linewidths=1, linecolor='white', square=False, ax=ax)
    ax.set_title(title, fontsize=12, pad=10)
    return fig


def draw_signal_comparison(v_orig, v_expand, title, color):
    """绘制信号拉伸前后的波形对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    # 原始信号
    ax1.plot(v_orig, marker='o', linestyle='--', color=color, alpha=0.7)
    ax1.set_title(f"原始信号 ({len(v_orig)}维)")
    ax1.grid(True, alpha=0.3)
    # 扩张信号
    ax2.plot(v_expand, marker='s', linestyle='-', color=color)
    ax2.set_title(f"STP扩张后 ({len(v_expand)}维)")
    ax2.grid(True, alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    return fig


# ================= 5. 页面主逻辑 =================

st.title("🧬 STP 跨维数生物特征融合演示系统")
st.caption("Designed for Academic Demonstration | 基于程代展教授 STP 理论框架")

mode = st.radio("选择演示模式", ["Mode A: 跨维矩阵乘法 (系统演化)", "Mode B: 多模态特征融合 (广义加法)"],
                horizontal=True)

st.markdown("---")

# ================= Mode A: 跨维矩阵乘法 =================
if "Mode A" in mode:
    st.header("✖️ 跨维数矩阵乘法 (STP Product)")
    st.markdown("演示如何解决 $A_{m \\times n} \\times B_{p \\times q}$ 当 $n \\neq p$ 时的运算问题。")

    # --- 参数设置 ---
    with st.container():
        st.subheader("1. 定义矩阵维度")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.markdown("### 矩阵 A")
            m = st.number_input("行数 (m)", 1, 5, 2, key='ma')
            n = st.number_input("列数 (n)", 1, 5, 2, key='na')
        with c2:
            st.markdown("### 矩阵 B")
            p = st.number_input("行数 (p)", 1, 5, 3, key='pb')
            q = st.number_input("列数 (q)", 1, 5, 2, key='qb')
        with c3:
            st.info("💡 **维度状态**")
            lcm_val = get_lcm(n, p)
            if n == p:
                st.success(f"维度匹配 ($n=p={n}$)，标准乘法。")
            else:
                st.warning(f"维度冲突 ($n={n} \\neq p={p}$)。\n需引入 STP，最小公倍数 $L={lcm_val}$。")

        if st.button("🎲 生成随机矩阵并计算", type="primary"):
            st.session_state.A = np.random.randint(1, 5, (m, n))
            st.session_state.B = np.random.randint(1, 5, (p, q))
            # 强制更新标记，防止使用旧维度的矩阵
            st.session_state.dims = (m, n, p, q)

    # 检查 session_state 是否存在或维度是否匹配
    if 'A' not in st.session_state or 'dims' not in st.session_state or st.session_state.dims != (m, n, p, q):
        st.session_state.A = np.random.randint(1, 5, (m, n))
        st.session_state.B = np.random.randint(1, 5, (p, q))
        st.session_state.dims = (m, n, p, q)

    A, B = st.session_state.A, st.session_state.B
    res = stp_product_stepwise(A, B)

    st.divider()

    # --- 理论推导 ---
    # ... (前面的代码保持不变) ...

    # --- 2. 理论推导与扩张过程 (Deep Dive) ---
    st.subheader("2. 核心机制：基于克罗内克积的扩张")

    # 拆分布局：左边讲原理，右边看图
    exp_c1, exp_c2 = st.columns([1.2, 1])

    with exp_c1:
        st.info("🤔 核心问题：为什么要做扩张？")
        st.markdown(f"""
            我们的目标是让 A 的列 ($n={n}$) 和 B 的行 ($p={p}$) 咬合。
            唯一的办法是把它们都映射到一个**公共的高维空间**，其维度为 $L = \\text{{LCM}}({n}, {p}) = {res['LCM']}$。
            """)

        st.markdown("#### 🔧 操作工具：单位矩阵 (Identity Matrix)")
        st.markdown(f"""
            为了“无损”地放大矩阵，我们使用 **单位矩阵 ($I_k$)** 作为扩张算子。
            它对角线为1，其余为0。

            在此次运算中，我们需要两个特定的单位矩阵：
            1. **用于 A 的算子 ($I_{{{res['alpha']}}}$)**: {res['alpha']} 维单位矩阵
            2. **用于 B 的算子 ($I_{{{res['beta']}}}$)**: {res['beta']} 维单位矩阵
            """)

        # 动态展示单位矩阵的样子
        if res['alpha'] > 1:
            I_a_disp = np.eye(res['alpha'], dtype=int)
            st.latex(rf"I_{{{res['alpha']}}} = " +
                     r"\begin{bmatrix} " +
                     r" \\ ".join([" & ".join(map(str, row)) for row in I_a_disp]) +
                     r" \end{bmatrix}")
        else:
            st.markdown(f"*注：A 不需要扩张 (因子为1)*")

        st.markdown("#### ⚡ 扩张操作：右克罗内克积 (Right Kronecker Product)")
        st.markdown(r"""
            扩张公式为：$A' = A \otimes I_k$。

            **这不仅仅是复制！** 它的物理动作是：
            把 A 中的**每一个元素** $a_{ij}$，都替换成一个 **对角块** $a_{ij} \times I_k$。
            """)

        # 举例说明
        example_val = A[0, 0]
        st.markdown(f"""
            > **举个栗子 🌰**：
            > 假设 A 的第一个元素是 **{example_val}**。
            > 在扩张后的矩阵 A' 中，这个 **{example_val}** 会变成一个 **{res['alpha']}x{res['alpha']}** 的小方块：
            """)

        # 构造一个小的 LaTeX 矩阵展示这个块
        block_content = r" \\ ".join(
            [" & ".join([str(example_val) if i == j else "0" for j in range(res['alpha'])]) for i in
             range(res['alpha'])])
        st.latex(
            rf"{example_val} \xrightarrow{{\otimes I_{{{res['alpha']}}}}} \begin{{bmatrix}} {block_content} \end{{bmatrix}}")

        st.success("""
            **为什么要这样？**
            使用单位矩阵 $I$ 而不是全1矩阵，是为了保持**稀疏性**和**线性独立性**。
            这保证了我们只是改变了“分辨率”（Dimension），而没有改变数据的“内容”（Structure）。
            """)

    with exp_c2:
        st.markdown("#### 👁️ 视觉验证")
        expand_tabs = st.tabs(["查看 A 的扩张细节", "查看 B 的扩张细节"])

        with expand_tabs[0]:
            st.write(f"**原始 A ({m}x{n})**")
            st.pyplot(draw_heatmap(A, "Original A", "Purples"))

            st.write("⬇️ **扩张后** (注意看数字是如何沿对角线排列的)")
            st.write(f"**扩张 A' = A ⊗ I_{res['alpha']}**")
            st.pyplot(draw_heatmap(res['A_kron'], f"Expanded A' ({res['A_kron'].shape})", "Purples"))

        with expand_tabs[1]:
            st.write(f"**原始 B ({p}x{q})**")
            st.pyplot(draw_heatmap(B, "Original B", "Oranges"))

            st.write("⬇️ **扩张后**")
            st.write(f"**扩张 B' = B ⊗ I_{res['beta']}**")
            st.pyplot(draw_heatmap(res['B_kron'], f"Expanded B' ({res['B_kron'].shape})", "Oranges"))

    st.divider()

    st.subheader("3. 最终结果")
    st.latex(r"Result = A \ltimes B = A' \times B'")

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        st.pyplot(draw_heatmap(res['Result'], "STP Result", "Greens"))
    with rc2:
        st.markdown(f"**结果维度:** ${res['Result'].shape[0]} \\times {res['Result'].shape[1]}$")

# ================= Mode B: 特征融合 =================
# ================= Mode B: 广义加法与特征融合 =================
elif "Mode B" in mode:
    # 子导航栏
    sub_mode = st.radio("Mode B 功能选择",
                        ["1. 基础理论：跨维矩阵加法 (原理演示)", "2. 应用场景：多模态特征融合 (LUTBIO案例)"],
                        horizontal=True)

    st.divider()

    # --- 子模块 1: 基础矩阵加法原理 ---
    if "1. 基础理论" in sub_mode:
        st.header("➕ 跨维矩阵加法 (STP Generalized Addition)")
        st.markdown("""
        **核心问题：** 传统矩阵加法要求 $A, B$ 维度完全一致。
        **STP 解决方案：** 利用 **Kronecker 积** 将矩阵“广播”到最小公倍数维度，实现跨维叠加。
        """)

        # 1. 参数设置
        with st.container():
            c1, c2, c3 = st.columns([1, 1, 1.5])
            with c1:
                st.markdown("### 矩阵 A")
                ma = st.number_input("行数 (m)", 1, 5, 2, key='ma_add')
                na = st.number_input("列数 (n)", 1, 5, 2, key='na_add')
            with c2:
                st.markdown("### 矩阵 B")
                mb = st.number_input("行数 (p)", 1, 5, 3, key='mb_add')
                nb = st.number_input("列数 (q)", 1, 5, 2, key='nb_add')
            with c3:
                st.info("💡 **维度分析**")
                lcm_row = get_lcm(ma, mb)
                lcm_col = get_lcm(na, nb)
                st.write(f"目标行数 (LCM): **{lcm_row}**")
                st.write(f"目标列数 (LCM): **{lcm_col}**")

            if st.button("🎲 生成随机矩阵 A 和 B", key="btn_gen_add"):
                st.session_state.A_add = np.random.randint(1, 10, (ma, na))
                st.session_state.B_add = np.random.randint(1, 10, (mb, nb))
                st.session_state.dims_add = (ma, na, mb, nb)

        # 初始化与校验
        if 'A_add' not in st.session_state or 'dims_add' not in st.session_state or st.session_state.dims_add != (
        ma, na, mb, nb):
            st.session_state.A_add = np.random.randint(1, 10, (ma, na))
            st.session_state.B_add = np.random.randint(1, 10, (mb, nb))
            st.session_state.dims_add = (ma, na, mb, nb)

        A, B = st.session_state.A_add, st.session_state.B_add

        # --- 计算逻辑 (局部定义，保持整洁) ---
        alpha_r, alpha_c = lcm_row // ma, lcm_col // na
        beta_r, beta_c = lcm_row // mb, lcm_col // nb

        # 使用全1矩阵进行广播扩张
        # 解释：加法通常意味着能量或信息的叠加，所以用全1矩阵相当于把一个像素点放大成一个色块
        J_A = np.ones((alpha_r, alpha_c), dtype=int)
        J_B = np.ones((beta_r, beta_c), dtype=int)

        A_exp = np.kron(A, J_A)
        B_exp = np.kron(B, J_B)
        Res_add = A_exp + B_exp

        # --- 2. 理论解释 ---
        st.subheader("🧐 扩张原理：全 1 矩阵广播 (Broadcasting)")
        t_theory, t_vis = st.columns([1, 1.5])

        with t_theory:
            st.markdown(f"""
            与乘法使用**单位矩阵 ($I$)** 不同，跨维加法通常使用 **全 1 矩阵 ($\mathbf{{1}}$)** 进行扩张。

            **为什么要用全 1 矩阵？**
            * 物理意义类似于 **“图像缩放 (Nearest Neighbor Resize)”**。
            * 我们把 A 中的每一个数值 $a_{{ij}}$，复制成一个 ${alpha_r} \\times {alpha_c}$ 的色块。
            * 这样保证了信息铺满整个空间，而不是像单位矩阵那样留下大量 0。

            **数学公式：**
            $$A' = A \otimes \mathbf{{1}}_{{{alpha_r} \\times {alpha_c}}}$$
            $$B' = B \otimes \mathbf{{1}}_{{{beta_r} \\times {beta_c}}}$$
            """)

            # 举例
            ex_val = A[0, 0]
            st.markdown(f"> **微观示例**：\n> 元素 **{ex_val}** 被扩张为：")
            block = np.full((alpha_r, alpha_c), ex_val)
            st.code(str(block).replace('[', '').replace(']', ''), language=None)

        with t_vis:
            tab_a, tab_b = st.tabs(["观察 A 的扩张", "观察 B 的扩张"])
            with tab_a:
                c_a1, c_a2 = st.columns(2)
                with c_a1: st.pyplot(draw_heatmap(A, f"原始 A ({ma}x{na})", "Blues"))
                with c_a2: st.pyplot(draw_heatmap(A_exp, f"广播后 A' ({lcm_row}x{lcm_col})", "Blues"))
            with tab_b:
                c_b1, c_b2 = st.columns(2)
                with c_b1: st.pyplot(draw_heatmap(B, f"原始 B ({mb}x{nb})", "Oranges"))
                with c_b2: st.pyplot(draw_heatmap(B_exp, f"广播后 B' ({lcm_row}x{lcm_col})", "Oranges"))

        st.divider()
        st.subheader("🏁 加法结果")
        st.latex(r"Result = (A \otimes \mathbf{1}) + (B \otimes \mathbf{1})")

        # 结果展示
        final_c1, final_c2 = st.columns([2, 1])
        with final_c1:
            st.pyplot(draw_heatmap(Res_add, "STP 加法结果", "Reds"))
        with final_c2:
            st.success("""
            **✅ 结果解读：**
            你看，原本风马牛不相及的两个矩阵，
            现在每一个位置都实现了精确的数值叠加。
            这就是 STP 处理异构数据的能力。
            """)

    # --- 子模块 2: 原有的生物特征融合案例 ---
    elif "2. 应用场景" in sub_mode:
        st.header("🧬 应用场景：LUTBIO 指纹与人脸融合")
        st.caption("基于前述矩阵加法原理，针对特征向量 (Vector) 的特殊应用")

        # ... (这里保留你之前 Mode B 的代码，只稍微调整缩进) ...
        c1, c2 = st.columns(2)
        with c1:
            dim_face = st.slider("人脸维度", 2, 20, 4)
        with c2:
            dim_finger = st.slider("指纹维度", 2, 20, 3)

        if st.button("🔄 刷新特征", key="btn_bio_ref"):
            st.session_state.v_face = np.random.randint(10, 50, (dim_face, 1))
            st.session_state.v_finger = np.random.randint(1, 10, (dim_finger, 1))

        if 'v_face' not in st.session_state or st.session_state.v_face.shape[0] != dim_face:
            st.session_state.v_face = np.random.randint(10, 50, (dim_face, 1))
            st.session_state.v_finger = np.random.randint(1, 10, (dim_finger, 1))

        res_add = stp_addition_stepwise(st.session_state.v_face, st.session_state.v_finger)

        st.subheader("📈 信号对齐视角")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.pyplot(draw_signal_comparison(st.session_state.v_face, res_add['V1_kron'], "人脸", "blue"))
        with c_s2:
            st.pyplot(draw_signal_comparison(st.session_state.v_finger, res_add['V2_kron'], "指纹", "orange"))

        st.subheader("🧮 融合结果")
        c_m1, c_m2, c_eq, c_m3 = st.columns([1, 1, 0.2, 1])
        with c_m1:
            st.pyplot(draw_heatmap(res_add['V1_kron'], "Face'", "Blues", False))
        with c_m2:
            st.pyplot(draw_heatmap(res_add['V2_kron'], "Finger'", "Oranges", False))
        with c_eq:
            st.markdown("### +")
        with c_m3:
            st.pyplot(draw_heatmap(res_add['Result'], "Fused", "Reds"))