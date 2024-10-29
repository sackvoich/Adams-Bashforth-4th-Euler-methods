import numpy as np
import matplotlib.pyplot as plt

# Определение правой части уравнения y' = f(x, y)
def f(x, y):
    return (1 + x) * np.exp(-y)

# Аналитическое решение
def exact_solution(x):
    return np.log((x**2 + 2*x + 2) / 2)

# Метод Эйлера для вычисления начальных значений
def euler_step(x, y, h):
    return y + h * f(x, y)

# Метод Адамса-Башфорта 4-го порядка
def adams_bashforth(x0, y0, x_end, h):
    n = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n)
    y = np.zeros(n)
    y[0] = y0

    # Используем метод Эйлера для первых 4 шагов
    for i in range(3):
        y[i+1] = euler_step(x[i], y[i], h)
    
    # Далее используем метод Адамса-Башфорта
    for i in range(3, n - 1):
        f1 = f(x[i], y[i])
        f2 = f(x[i-1], y[i-1])
        f3 = f(x[i-2], y[i-2])
        f4 = f(x[i-3], y[i-3])
        
        y[i+1] = y[i] + h * (55*f1 - 59*f2 + 37*f3 - 9*f4) / 24
    
    return x, y

# Основная часть программы
x0 = 0.4
y0 = 1
x_end = 1.9
h = 0.1

# Численное решение методом Адамса
x_numerical, y_numerical = adams_bashforth(x0, y0, x_end, h)

# Аналитическое решение
y_exact = exact_solution(x_numerical)

# Вывод результатов в виде таблицы
print("    x      y_num      y_exact       error        Method")
print("------------------------------------------------------------")
for i in range(len(x_numerical)):
    error = abs(y_numerical[i] - y_exact[i])
    method = "Euler" if i < 4 else "Adams"
    print(f"{x_numerical[i]:6.2f}  {y_numerical[i]:10.3f}  {y_exact[i]:10.3f}  {error:10.3e}  {method:>10}")

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(x_numerical, y_numerical, 'b-', label='Численное решение (Адамс)')
plt.plot(x_numerical, y_exact, 'r--', label='Аналитическое решение')
plt.plot(x_numerical[:4], y_numerical[:4], 'go', markersize=8, label='Метод Эйлера')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Сравнение численного и аналитического решений')
plt.grid(True)

# Добавление аннотаций для точек Эйлера
for i in range(4):
    plt.annotate(f'({x_numerical[i]:.1f}, {y_numerical[i]:.4f})', 
                 (x_numerical[i], y_numerical[i]), 
                 xytext=(5, 5), textcoords='offset points')

plt.show()