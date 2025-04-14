import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Simulação física do pistão
class SimulacaoPistao:
    def __init__(self, num_particulas, massa_pistao, area_pistao, velocidade_particulas, forca_estocastica_std=0.5):
        self.num_particulas = num_particulas
        self.massa_pistao = massa_pistao
        self.area_pistao = area_pistao
        self.velocidade_particulas = velocidade_particulas
        self.forca_estocastica_std = forca_estocastica_std
        self.gravidade = 9.8
        self.k_b = 1.38e-23
        self.n = 1.0
        self.R = 8.314

        self.posicoes = np.random.rand(num_particulas) * 0.9
        self.velocidades = np.random.choice([-1, 1], num_particulas) * velocidade_particulas
        self.posicao_pistao = 1.0
        self.dt = 0.001
        self.tempo = []
        self.dados_pressao = []
        self.atualizar_temperatura()

    def atualizar_temperatura(self):
        energia_cinetica = 0.5 * np.mean(self.velocidades**2)
        self.temperatura = (2 / 3) * (energia_cinetica * self.massa_pistao) / self.R

    def calcular_pressao(self):
        volume = self.area_pistao * self.posicao_pistao
        if volume <= 0:
            volume = 1e-6
        pressao = (self.n * self.R * self.temperatura) / volume
        return pressao

    def passo_simulacao(self):
        forca_estocastica = np.random.normal(0, self.forca_estocastica_std, self.num_particulas)
        self.velocidades += forca_estocastica * self.dt
        self.posicoes += self.velocidades * self.dt

        colisao_pistao = self.posicoes >= self.posicao_pistao
        self.posicoes[colisao_pistao] = self.posicao_pistao
        self.velocidades[colisao_pistao] *= -1

        colisao_base = self.posicoes <= 0
        self.posicoes[colisao_base] = 0
        self.velocidades[colisao_base] *= -1

        self.atualizar_temperatura()
        pressao = self.calcular_pressao()

        forca_no_pistao = pressao * self.area_pistao
        forca_liquida = forca_no_pistao - (self.massa_pistao * self.gravidade)
        self.posicao_pistao += (forca_liquida / self.massa_pistao) * self.dt

        if self.posicao_pistao < 0.05:
            self.posicao_pistao = 0.05

        if len(self.tempo) == 0:
            self.tempo.append(0)
        else:
            self.tempo.append(self.tempo[-1] + self.dt)
        self.dados_pressao.append(pressao)

        return pressao

# Interface gráfica e animação
class AplicacaoSimulacao:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulação de Pistão com Zoom e Equilíbrio")

        self.num_particulas = 100
        self.massa_pistao = 3.0
        self.area_pistao = 0.1
        self.velocidade_particulas = 4.8

        self.simulacao = SimulacaoPistao(
            self.num_particulas,
            self.massa_pistao,
            self.area_pistao,
            self.velocidade_particulas
        )

        # Criando a figura com 3 eixos: Simulação, Gráfico Zoom, Gráfico Equilíbrio
        self.fig, (self.ax_simulacao, self.ax_grafico_zoom, self.ax_grafico_equilibrio) = plt.subplots(1, 3, figsize=(18, 5))

        # Configuração do eixo da simulação
        self.ax_simulacao.set_xlim(0, 1)
        self.ax_simulacao.set_ylim(0, 1.2)
        self.ax_simulacao.set_title("Simulação")
        self.particles, = self.ax_simulacao.plot([], [], 'bo', markersize=2)
        self.piston, = self.ax_simulacao.plot([0, 1], [self.simulacao.posicao_pistao, self.simulacao.posicao_pistao], 'r-', lw=2)
        self.block = self.ax_simulacao.add_patch(plt.Rectangle((0.4, self.simulacao.posicao_pistao), 0.2, 0.05, color='gray'))

        # Configuração do gráfico de zoom
        self.ax_grafico_zoom.set_title("Zoom: Pressão em Função do Tempo")
        self.ax_grafico_zoom.set_xlabel("Tempo (s)")
        self.ax_grafico_zoom.set_ylabel("Pressão (Pa)")
        self.ax_grafico_zoom.grid()
        self.linha_pressao_zoom, = self.ax_grafico_zoom.plot([], [], label="Pressão (Zoom)", color='blue')

        # Configuração do gráfico de equilíbrio
        self.ax_grafico_equilibrio.set_title("Pressão em função do tempo")
        self.ax_grafico_equilibrio.set_xlabel("Tempo (s)")
        self.ax_grafico_equilibrio.set_ylabel("Pressão (Pa)")
        self.ax_grafico_equilibrio.grid()
        self.linha_pressao_equilibrio, = self.ax_grafico_equilibrio.plot([], [], label="Pressão (Equilíbrio)", color='green')

        # Integração do canvas matplotlib com tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.num_particulas_visiveis = min(50, self.num_particulas)

        # Configura a animação
        self.ani = FuncAnimation(self.fig, self.atualizar_animacao, interval=20, blit=False)

    def atualizar_animacao(self, frame):
        passos_por_frame = 10
        for _ in range(passos_por_frame):
            self.simulacao.passo_simulacao()

        # Atualiza os dados do gráfico de zoom (últimos 500 pontos)
        max_pontos_grafico = 500
        self.linha_pressao_zoom.set_data(
            self.simulacao.tempo[-max_pontos_grafico:], 
            self.simulacao.dados_pressao[-max_pontos_grafico:]
        )
        self.ax_grafico_zoom.relim()
        self.ax_grafico_zoom.autoscale_view()

        # Atualiza os dados do gráfico de equilíbrio (todos os pontos)
        self.linha_pressao_equilibrio.set_data(
            self.simulacao.tempo, 
            self.simulacao.dados_pressao
        )
        self.ax_grafico_equilibrio.relim()
        self.ax_grafico_equilibrio.autoscale_view()

        # Atualiza a posição das partículas e do pistão
        self.particles.set_data(
            np.random.rand(self.num_particulas_visiveis), 
            self.simulacao.posicoes[:self.num_particulas_visiveis]
        )
        self.piston.set_ydata([self.simulacao.posicao_pistao, self.simulacao.posicao_pistao])
        self.block.set_xy((0.4, self.simulacao.posicao_pistao))

        self.canvas.draw()
        return self.particles, self.piston, self.block

# Código principal
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacaoSimulacao(root)
    root.mainloop()
