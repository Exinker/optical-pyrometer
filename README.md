
# Optical Pyrometer

## Излучение черного тела

Для абсолютно чёрного тела **спектральная плотность излучения** на единицу площади ($W/m^{3}$) можно рассчитать по формуле:
$$I_{\lambda} = \frac{2 \pi h c^{2}}{\lambda^{5}}\frac{1}{e^{hc / \lambda k T} - 1},$$
где $T$ - температура ($^{\circ}K$), $h$ - постоянная Планка ($J \cdot c$), $k$ - постоянная Больцмана ($J / K$), $c$ - скорость света ($m/c$).

![density-radiation](./img/density-radiation.png)
*Рис. 1. Спектральная плотность излучения при заданной температуре*

Тогда **излучение** на единицу площади ($W/m^{2}$) можно определить по формуле:
$$I = \int_{\lambda_{min}}^{\lambda_{max}}{I_{\lambda} \cdot d\lambda},$$
где $[\lambda_{min}; \lambda_{max}]$ - диапазон чувствительности детектора.


## Параметры системы измерения
Пусть требуется измерить температуру объекта в диапазоне $T \in [T_{1}, T_{2}]$.

### Разрешение АЦП
Разрешение АЦП определяет шаг дискретизации измеренного **излучения**:
$$\Delta{I} = \frac{I_{2} - I_{1}}{2^{n} - 1},$$
где $n$ - разрешение АЦП, $I_{1}$ и $I_{2}$ излучение объекта при температуре $T_{1}$ и $T_{2}$ соответственно.

При этом, если требуется погрешность измерения температуры не более $10^{\circ}C$, то шаг дискретизации $\Delta{I}$ должен быть меньше, чем изменение излучения при изменение температуры на $10^{\circ}C$:
$$\Delta{I} < I_{T_{1} + 10} - I_{T_{1}}.$$

![irradiance](./img/irradiance-6.png)
*Рис. 2. Зарегистрированный дискретный сигнал (синим) при 6 битном АЦП (слева) и полученная погрешность измерения температуры (справа). Излучение абсолютно черного тела (черным) получено при изменение температуры с 500 до 1250 $^{\circ}C$*


![irradiance](./img/irradiance-12.png)
*Рис. 3. Зарегистрированный дискретный сигнал (синим) при 12 битном АЦП (слева) и полученная погрешность измерения температуры (справа). Излучение абсолютно черного тела (черным) получено при изменение температуры с $500$ до $1250^{\circ}C$*
