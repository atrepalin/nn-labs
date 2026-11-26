import numpy as np
from manim import *


class SelfAttentionRussian(Scene):
    def construct(self):
        self.camera.background_color = "#10141F"

        words = ["мама", "мыла", "раму"]

        # Цвета слов НЕ кодируют Q, K или V.
        WORD_COLOR = WHITE

        # Отдельная цветовая система для механизмов внимания.
        Q_COLOR = BLUE_C
        K_COLOR = TEAL_C
        V_COLOR = YELLOW

        # Цвета строк соответствуют позициям слов.
        ROW_COLORS = [BLUE_C, GOLD, GREEN_C]

        # K: какую информацию предлагает слово.
        #
        # мама: сильный признак субъекта
        # мыла: сильный признак действия + слабая подсказка
        #       о субъекте (согласование в женском роде)
        # раму: сильный признак объекта
        K = np.array(
            [
                [1.00, 0.00, 0.00],
                [0.15, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ]
        )

        # Q: какую информацию ищет текущая позиция.
        #
        # мама: ищет субъекта
        # мыла: ищет субъекта и действие
        # раму: ищет действие и объект
        Q = np.array(
            [
                [1.00, 0.00, 0.00],
                [0.70, 1.00, 0.00],
                [0.00, 0.25, 1.00],
            ]
        )

        # V: какая информация передаётся дальше.
        #
        # Глагол несёт действие и немного грамматической
        # информации о субъекте.
        V = np.array(
            [
                [1.00, 0.00, 0.00],
                [0.15, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ]
        )

        # ------------------------------------------------
        # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
        # ------------------------------------------------

        def header(text):
            return MarkupText(
                text,
                font_size=28,
                weight=BOLD,
                color=WHITE,
            ).to_edge(UP, buff=0.22)

        def footer(text):
            return MarkupText(
                text,
                font_size=15,
                color=GRAY_A,
            ).to_edge(DOWN, buff=0.16)

        def vector_tex(vec, color):
            result = MathTex(
                r"\begin{bmatrix}"
                + r"\\ ".join(f"{x:.2f}" for x in vec)
                + r"\end{bmatrix}",
                font_size=27,
                color=color,
            )
            return result

        def softmax(x):
            x = np.array(x, dtype=float)
            e = np.exp(x - np.max(x))
            return e / e.sum()

        def make_row_highlight(mobject, color, opacity=0.16):
            return SurroundingRectangle(
                mobject,
                color=color,
                fill_color=color,
                fill_opacity=opacity,
                stroke_color=color,
                stroke_width=1.5,
                buff=0.09,
            )

        def make_matrix_labels(matrix, labels, direction="row"):
            if direction == "row":
                result = VGroup()
                for i, label in enumerate(labels):
                    entry = matrix.get_entries()[i * 3]
                    text = Text(
                        label,
                        font_size=19,
                        color=ROW_COLORS[i],
                    )
                    text.next_to(entry, LEFT, buff=0.65)
                    result.add(text)
                return result

            result = VGroup()
            for j, label in enumerate(labels):
                entry = matrix.get_entries()[j]
                text = Text(
                    label,
                    font_size=19,
                    color=ROW_COLORS[j],
                )
                text.next_to(entry, UP, buff=0.35)
                result.add(text)
            return result

        # Scores[i, j] = Q_i · K_j
        scores = Q @ K.T

        # Softmax по всем позициям каждой строки.
        weights = np.zeros((3, 3))
        for i in range(3):
            weights[i] = softmax(scores[i])

        # Итоговое смешивание Value.
        output = weights @ V

        # =================================================
        # СЦЕНА 1. Знакомство с Q, K и V
        # =================================================

        title = header("Self-attention: от грамматики к весам")

        sentence = VGroup(
            *[Text(w, font_size=42, color=WORD_COLOR) for w in words]
        ).arrange(RIGHT, buff=0.85)

        sentence.move_to(UP * 1.95)

        subtitle = Text(
            "Три слова. Никаких embedding.",
            font_size=22,
            color=GRAY_A,
        ).next_to(sentence, DOWN, buff=0.3)

        self.play(Write(title))
        self.play(
            LaggedStart(
                *[FadeIn(w, shift=UP * 0.2) for w in sentence],
                lag_ratio=0.2,
            )
        )
        self.play(FadeIn(subtitle))

        descriptions = VGroup(
            Text("Q — что ищет текущая позиция", font_size=21, color=Q_COLOR),
            Text("K — какую информацию предлагает слово", font_size=21, color=K_COLOR),
            Text("V — какая информация передаётся дальше", font_size=21, color=V_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)

        descriptions.move_to(DOWN * 0.55)

        self.play(FadeIn(descriptions))
        self.wait(1.5)

        self.play(
            FadeOut(sentence),
            FadeOut(subtitle),
            FadeOut(descriptions),
        )

        # =================================================
        # СЦЕНА 2. Векторы Q и K
        # =================================================

        self.play(Transform(title, header("Шаг 1. Задаём грамматические признаки")))

        feature_legend = VGroup(
            Text("СУБЪЕКТ", font_size=17, color=ROW_COLORS[0]),
            Text("ДЕЙСТВИЕ", font_size=17, color=ROW_COLORS[1]),
            Text("ОБЪЕКТ", font_size=17, color=ROW_COLORS[2]),
        ).arrange(RIGHT, buff=0.55)
        feature_legend.move_to(UP * 2.1)

        self.play(FadeIn(feature_legend))

        # Показываем отдельно Q, K, V.
        vector_rows = VGroup()

        for i, word in enumerate(words):
            word_label = Text(
                word,
                font_size=23,
                color=WORD_COLOR,
            )

            q_label = MathTex("Q", font_size=25, color=Q_COLOR)
            q_vec = vector_tex(Q[i], Q_COLOR)

            k_label = MathTex("K", font_size=25, color=K_COLOR)
            k_vec = vector_tex(K[i], K_COLOR)

            v_label = MathTex("V", font_size=25, color=V_COLOR)
            v_vec = vector_tex(V[i], V_COLOR)

            row = VGroup(
                word_label,
                q_label,
                q_vec,
                k_label,
                k_vec,
                v_label,
                v_vec,
            ).arrange(RIGHT, buff=0.23)

            vector_rows.add(row)

        vector_rows.arrange(DOWN, buff=0.37)
        vector_rows.move_to(DOWN * 0.25)

        # Цветные рамки подчёркивают строку соответствующего слова.
        row_boxes = VGroup()
        for i, row in enumerate(vector_rows):
            row_boxes.add(
                SurroundingRectangle(
                    row,
                    color=ROW_COLORS[i],
                    stroke_width=1.5,
                    buff=0.12,
                )
            )

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        Create(row_boxes[i]),
                        FadeIn(vector_rows[i], shift=RIGHT * 0.15),
                    )
                    for i in range(3)
                ],
                lag_ratio=0.25,
            )
        )

        legend_note = footer("Q — синий, K — бирюзовый, V — жёлтый")
        self.play(FadeIn(legend_note))
        self.wait(2)

        self.play(
            FadeOut(feature_legend),
            FadeOut(vector_rows),
            FadeOut(row_boxes),
            FadeOut(legend_note),
        )

        # =================================================
        # СЦЕНА 3. Поэлементное произведение Q · K
        # =================================================

        self.play(Transform(title, header("Шаг 2. Как получается оценка внимания")))

        intro = Text(
            "Насколько Query слова «мыла» совпадает с Key слова «мама»?",
            font_size=19,
        ).move_to(UP * 2.0)

        self.play(FadeIn(intro))

        q_label = MathTex("Q_2 =", color=Q_COLOR, font_size=32)
        q_vec = vector_tex(Q[1], Q_COLOR)
        dot = MathTex(r"\cdot", font_size=32)
        k_label = MathTex("K_1 =", color=K_COLOR, font_size=32)
        k_vec = vector_tex(K[0], K_COLOR)

        vectors = VGroup(q_label, q_vec, dot, k_label, k_vec).arrange(RIGHT, buff=0.2)
        vectors.move_to(UP * 0.95)

        self.play(FadeIn(vectors))

        products = VGroup()
        for j in range(3):
            term = MathTex(
                f"{Q[1, j]:.2f}",
                r"\times",
                f"{K[0, j]:.2f}",
                "=",
                f"{Q[1, j] * K[0, j]:.2f}",
                font_size=28,
            )
            products.add(term)

        products.arrange(RIGHT, buff=0.32)
        products.move_to(DOWN * 0.15)

        for term in products:
            self.play(FadeIn(term, shift=UP * 0.1), run_time=0.35)
            self.play(Indicate(term, color=YELLOW), run_time=0.3)

        score_formula = MathTex(
            r"s_{21}=Q_2\cdot K_1",
            r"=0.70\cdot1.00+1.00\cdot0.00+0.00\cdot0.00",
            r"=0.70",
            font_size=24,
        ).arrange(DOWN, buff=0.2)
        score_formula.move_to(DOWN * 1.35)

        self.play(Write(score_formula))
        self.play(Circumscribe(score_formula[-1], color=YELLOW))

        note = footer(
            "Скалярное произведение складывает попарные произведения координат"
        )
        self.play(FadeIn(note))
        self.wait(2)

        self.play(
            FadeOut(intro),
            FadeOut(vectors),
            FadeOut(products),
            FadeOut(score_formula),
            FadeOut(note),
        )

        # =================================================
        # СЦЕНА 4. Матрица оценок QK^T
        # =================================================

        self.play(
            Transform(title, header("Шаг 3. Матрица совместимости Q · K<sup>T</sup>"))
        )

        score_matrix = Matrix(
            np.round(scores, 2).tolist(),
            element_to_mobject=lambda x: DecimalNumber(
                float(x), num_decimal_places=2, font_size=27
            ),
            h_buff=1.0,
            v_buff=0.72,
        ).scale(0.95)

        score_matrix.move_to(LEFT * 1.1 + UP * 0.15)

        # Цветные подписи слов по строкам и столбцам.
        row_labels = VGroup()
        for i, word in enumerate(words):
            entry = score_matrix.get_entries()[i * 3]
            label = Text(word, font_size=19, color=ROW_COLORS[i])
            label.next_to(entry, LEFT, buff=0.6)
            row_labels.add(label)

        col_labels = VGroup()
        for j, word in enumerate(words):
            entry = score_matrix.get_entries()[j]
            label = Text(word, font_size=19, color=ROW_COLORS[j])
            label.next_to(entry, UP, buff=0.55)
            col_labels.add(label)

        row_boxes = VGroup()
        for i in range(3):
            row_entries = VGroup(*score_matrix.get_entries()[i * 3 : (i + 1) * 3])
            row_boxes.add(make_row_highlight(row_entries, ROW_COLORS[i], 0.12))

        matrix_frame = SurroundingRectangle(score_matrix, color=GRAY_B, buff=0.1)

        side_note = VGroup(
            Text("Строка = Query", font_size=18, color=Q_COLOR),
            Text("Столбец = Key", font_size=18, color=K_COLOR),
            MathTex(r"S=QK^\top", font_size=31),
            Text(
                "Цвет рамки строки\nсоответствует слову",
                font_size=17,
                color=GRAY_A,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)

        side_note.move_to(RIGHT * 3.0 + UP * 0.2)

        self.play(
            FadeIn(score_matrix),
            FadeIn(row_labels),
            FadeIn(col_labels),
            Create(matrix_frame),
        )

        # Выделяем строки последовательно, сохраняя цветовую связь
        # между словом, Query и соответствующей строкой матрицы.
        for i in range(3):
            self.play(
                Create(row_boxes[i]),
                Indicate(row_labels[i], color=ROW_COLORS[i]),
                run_time=0.55,
            )
            for j in range(3):
                self.play(
                    Indicate(
                        score_matrix.get_entries()[i * 3 + j],
                        color=ROW_COLORS[i],
                    ),
                    run_time=0.18,
                )

        self.play(FadeIn(side_note))
        self.wait(2)

        self.play(
            FadeOut(score_matrix),
            FadeOut(row_labels),
            FadeOut(col_labels),
            FadeOut(row_boxes),
            FadeOut(matrix_frame),
            FadeOut(side_note),
        )

        # =================================================
        # СЦЕНА 5. Softmax
        # =================================================

        self.play(Transform(title, header("Шаг 4. Softmax превращает оценки в веса")))

        weight_matrix = MobjectMatrix(
            [
                [
                    DecimalNumber(
                        weights[i, j],
                        num_decimal_places=2,
                        font_size=26,
                    )
                    for j in range(3)
                ]
                for i in range(3)
            ],
            h_buff=1.05,
            v_buff=0.75,
        ).scale(0.95)

        weight_matrix.move_to(LEFT * 1.1 + UP * 0.2)

        weight_rows = VGroup()
        for i, word in enumerate(words):
            # Используем геометрию числовых элементов для точного
            # расположения подписи строки.
            row_group = VGroup(*weight_matrix.get_entries()[i * 3 : (i + 1) * 3])
            label = Text(word, font_size=19, color=ROW_COLORS[i])
            label.next_to(row_group, LEFT, buff=0.45)
            weight_rows.add(label)

        weight_cols = VGroup()
        for j, word in enumerate(words):
            entry = weight_matrix.get_entries()[j]
            label = Text(word, font_size=19, color=ROW_COLORS[j])
            label.next_to(entry, UP, buff=0.25)
            weight_cols.add(label)

        weight_boxes = VGroup()
        for i in range(3):
            row_group = VGroup(*weight_matrix.get_entries()[i * 3 : (i + 1) * 3])
            weight_boxes.add(make_row_highlight(row_group, ROW_COLORS[i], 0.12))

        softmax_formula = MathTex(
            r"\alpha_{ij}=" r"\frac{e^{s_{ij}}}" r"{\sum_{k=1}^{3}e^{s_{ik}}}",
            font_size=28,
        ).move_to(RIGHT * 3.0 + UP * 0.85)

        softmax_note = VGroup(
            Text("В каждой строке сумма весов = 1", font_size=18),
            Text("Все позиции доступны для внимания", font_size=16, color=GREEN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        softmax_note.next_to(softmax_formula, DOWN, buff=0.45)
        softmax_note.align_to(softmax_formula, LEFT)

        self.play(
            FadeIn(weight_matrix),
            FadeIn(weight_rows),
            FadeIn(weight_cols),
        )

        for i in range(3):
            self.play(Create(weight_boxes[i]), run_time=0.35)
            for j in range(3):
                entry = weight_matrix.get_entries()[i * 3 + j]
                self.play(
                    Indicate(entry, color=ROW_COLORS[i]),
                    run_time=0.2,
                )

        self.play(Write(softmax_formula), FadeIn(softmax_note))
        self.wait(2)

        self.play(
            FadeOut(weight_matrix),
            FadeOut(weight_rows),
            FadeOut(weight_cols),
            FadeOut(weight_boxes),
            FadeOut(softmax_formula),
            FadeOut(softmax_note),
        )

        # =================================================
        # СЦЕНА 6. Смешивание Value
        # =================================================

        self.play(Transform(title, header("Шаг 5. Взвешиваем Value")))

        value_heading = MathTex(
            r"O_i=\sum_{j=1}^{3}\alpha_{ij}V_j",
            font_size=34,
            color=V_COLOR,
        ).move_to(UP * 2)

        self.play(Write(value_heading))

        # Сначала показываем V каждого слова.
        value_rows = VGroup()
        for i, word in enumerate(words):
            label = Text(word, font_size=22, color=WORD_COLOR)
            v_label = MathTex("V", font_size=25, color=V_COLOR)
            v_vec = vector_tex(V[i], V_COLOR)

            row = VGroup(label, v_label, v_vec).arrange(RIGHT, buff=0.3)
            value_rows.add(row)

        value_rows.arrange(RIGHT, buff=0.65)
        value_rows.move_to(UP * 0.65)

        value_boxes = VGroup()
        for i, row in enumerate(value_rows):
            value_boxes.add(
                SurroundingRectangle(
                    row,
                    color=ROW_COLORS[i],
                    stroke_width=1.5,
                    buff=0.12,
                )
            )

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(Create(value_boxes[i]), FadeIn(value_rows[i]))
                    for i in range(3)
                ],
                lag_ratio=0.2,
            )
        )

        result_rows = VGroup()

        for i, word in enumerate(words):
            terms = [rf"{weights[i, j]:.2f}V_{j + 1}" for j in range(3)]

            expression = MathTex(
                rf"O_{{{i + 1}}}=" + "+".join(terms),
                font_size=25,
                color=V_COLOR,
            )

            label = Text(word, font_size=22, color=ROW_COLORS[i])
            row = VGroup(label, expression).arrange(RIGHT, buff=0.35)
            result_rows.add(row)

        result_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        result_rows.move_to(DOWN * 0.85)

        for i, row in enumerate(result_rows):
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.5)

            # Подсвечиваем Value всех трёх позиций.
            for j in range(3):
                self.play(
                    Indicate(value_boxes[j], color=YELLOW),
                    run_time=0.3,
                )

        final_note = footer(
            "Q · K<sup>T</sup> определяет веса внимания; V определяет передаваемую информацию"
        )

        self.play(FadeIn(final_note))
        self.wait(3)

        # =================================================
        # СЦЕНА 7. Финальный итог
        # =================================================

        self.play(
            FadeOut(value_heading),
            FadeOut(value_rows),
            FadeOut(value_boxes),
            FadeOut(result_rows),
            FadeOut(final_note),
            Transform(title, header("Что делает self-attention?")),
        )

        summary = VGroup(
            Text("Q", font_size=33, color=Q_COLOR),
            Text("что ищем", font_size=22),
            MathTex(r"\cdot", font_size=30),
            Text("K", font_size=33, color=K_COLOR),
            Text("что предлагаем", font_size=22),
            MathTex(r"\longrightarrow", font_size=30),
            Text("Softmax", font_size=24, color=YELLOW),
            MathTex(r"\longrightarrow", font_size=30),
            Text("V", font_size=33, color=V_COLOR),
            Text("что передаём", font_size=22),
        ).arrange(RIGHT, buff=0.17)

        summary.scale_to_fit_width(12.2)
        summary.move_to(UP * 0.8)

        causal_note = Text(
            "Каждое слово учитывает контекст всей последовательности.",
            font_size=22,
            color=GREEN_C,
        ).move_to(DOWN * 0.4)

        linguistic_note = Text(
            "Числа заданы вручную для демонстрации, а не обучены на данных.",
            font_size=17,
            color=GRAY_A,
        ).move_to(DOWN * 1.25)

        self.play(FadeIn(summary, shift=UP * 0.15))
        self.play(Write(causal_note))
        self.play(FadeIn(linguistic_note))
        self.wait(3)
