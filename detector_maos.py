import cv2
import mediapipe as mp

class DetectorMaos:
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5,
                 rastreio_confianca=0.5, cor_pontos=(0, 0, 255), cor_conexoes=(255, 255, 255)):
        # O Python exige algo aqui dentro. Use 'pass' se for preencher depois.

        """
        Função responsável por inicializar a classe.

        :param modo: Modo de captura da imagem. Se true, a detecção e rastreio serão feitos a todo momento;
        deixa muito travado. Se false, não são feitas a detecção e rastreio a todo momento; pode perder por
        alguns instantes as marcações, porém não trava

        :param max_maos: Quantidade máxima de mãos que podem ser detectadas.

        :param deteccao_confianca: Percentual da taxa de detecção da mão. Se for menor do que este limite, a detecção não ocorre.

        :param rastreio_confianca: Percentual da taxa de rastreio dos pontos da mão. Se for menor que este limite, o rastreio dos pontos não ocorre.

        :param cor_pontos: Cor dos Pontos

        :param cor_conexoes: Cor das Conexões

        """

        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes
        self.resultado = None

        # CORREÇÃO: Usando mediapipe.solutions diretamente para evitar o erro de atributo
        self.maos_mp = mp.solutions.hands
        self.maos = self.maos_mp.Hands(
            static_image_mode=self.modo,
            max_num_hands=self.max_maos,
            model_complexity=1,
            min_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca,
        )

        # Função para desenhar os pontos nas mãos

        self.desenho_mp = mp.solutions.drawing_utils

        # Configurações do desenho dos pontos

        self.desenho_config_pontos = self.desenho_mp.DrawingSpec(color=self.cor_pontos)

        # Configurações do desenho das conexões

        self.desenho_config_conexoes = self.desenho_mp.DrawingSpec(color=self.cor_conexoes)

    # Corrigido: Agora está fora do __init__ e com a sintaxe correta
    def encontrar_maos(self, imagem, desenho=True):

        """

        :param imagem: Imagem capturada
        :param desenho: Desenhar os pontos e as conexões na mão
        :return: Retorna a imagem com a detecção
        """

        # -- Converter a imagem de BGR para RGB --#

        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # -- Passar a imagem convertida para o detector --#

        self.resultado = self.maos.process(imagem_rgb)

        # Desenhar os pontos se houver detecção
        if self.resultado.multi_hand_landmarks:
            for pontos_maos in self.resultado.multi_hand_landmarks:
                if desenho:
                    self.desenho_mp.draw_landmarks(imagem, pontos_maos, self.maos_mp.HAND_CONNECTIONS,
                                                   self.desenho_config_pontos, self.desenho_config_conexoes)

        return imagem

        # --- Verificar se alguma mão não foi detectada --- #

        if self.resultado.multi_hand_landmarks:
            for pontos in self.resultado.multi_hand_landmarks:
                if desenho:
                    # -- Desenhar os pontos nas mãos detectada --#
                        self.desenho_mp.draw_landmarks(
                            imagem, # Imagem da Captura
                            pontos, # Pontos da Mão
                            self.mao_mp.HAND_CONNECTIONS, # Conexão entre os pontos
                            self.desenho_config_pontos, # Cor dos pontos
                            self.desenho_config_conexoes, # Cor da Conexoes
                        )
            return imagem


def main():
    cap = cv2.VideoCapture(0)

    # -- Instanciar a classe detector

    detector = DetectorMaos()

    while True:

        _,imagem = cap.read()

        imagem = cv2.flip(imagem, 1)

        # -- Detecção das mãos

        imagem = detector.encontrar_maos(imagem)

        cv2.imshow('Captura dos Gestos - TCC', imagem)




if __name__ == '__main__':
    main()