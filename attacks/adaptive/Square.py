from IPython import embed
from abc import abstractmethod
import torch
from tqdm.auto import tqdm
from IPython import embed
from attacks.Attack import Attack
import numpy as np


class Square(Attack):
    def __init__(self, model, model_config, attack_config):
        super().__init__(model, model_config, attack_config)

    # def attack_untargeted(self, x, y):
    #     dim = torch.prod(torch.tensor(x.shape[1:]))
    #
    #     def p_selection(step):
    #         step = int(step / self.attack_config["max_iter"] * 10000)
    #         if 10 < step <= 50:
    #             p = self.attack_config["p_init"] / 2
    #         elif 50 < step <= 200:
    #             p = self.attack_config["p_init"] / 4
    #         elif 200 < step <= 500:
    #             p = self.attack_config["p_init"] / 8
    #         elif 500 < step <= 1000:
    #             p = self.attack_config["p_init"] / 16
    #         elif 1000 < step <= 2000:
    #             p = self.attack_config["p_init"] / 32
    #         elif 2000 < step <= 4000:
    #             p = self.attack_config["p_init"] / 64
    #         elif 4000 < step <= 6000:
    #             p = self.attack_config["p_init"] / 128
    #         elif 6000 < step <= 8000:
    #             p = self.attack_config["p_init"] / 256
    #         elif 8000 < step <= 10000:
    #             p = self.attack_config["p_init"] / 512
    #         else:
    #             p = self.attack_config["p_init"]
    #         return p
    #
    #     def margin_loss(x, y):
    #         logits, is_cache = self.model(x)
    #         probs = torch.softmax(logits, dim=1)
    #         top_2_probs, top_2_classes = torch.topk(probs, 2)
    #         if top_2_classes[:, 0] != y:
    #             return 0, is_cache
    #         else:
    #             return top_2_probs[:, 0] - top_2_probs[:, 1], is_cache
    #
    #     # Initialize adversarial example
    #     pert = torch.tensor(np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
    #                                          size=[x.shape[0], x.shape[1], 1, x.shape[3]])).float().to(x.device)
    #     x_adv = torch.clamp(x + pert, 0, 1)
    #     loss, is_cache = margin_loss(x_adv, y)
    #
    #     pbar = tqdm(range(self.attack_config["max_iter"]))
    #     for t in pbar:
    #         x_adv_candidate = x_adv.clone()
    #         for _ in range(self.attack_config["num_squares"]):
    #             # pert = x_adv - x
    #             pert = x_adv_candidate - x
    #             s = int(min(max(torch.sqrt(p_selection(t) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
    #             center_h = torch.randint(0, x.shape[2] - s, size=(1,)).to(x.device)
    #             center_w = torch.randint(0, x.shape[3] - s, size=(1,)).to(x.device)
    #             x_window = x[:, :, center_h:center_h + s, center_w:center_w + s]
    #             x_adv_window = x_adv_candidate[:, :, center_h:center_h + s, center_w:center_w + s]
    #
    #             while torch.sum(
    #                     torch.abs(
    #                         torch.clamp(
    #                             x_window + pert[:, :, center_h:center_h + s, center_w:center_w + s], 0, 1
    #                         ) -
    #                         x_adv_window)
    #                     < 10 ** -7) == x_adv.shape[1] * s * s:
    #                 pert[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(
    #                     np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]], size=[x_adv.shape[1], 1, 1])).float().to(x_adv.device)
    #
    #             x_adv_candidate = torch.clamp(x + pert, 0, 1)
    #         new_loss, is_cache = margin_loss(x_adv_candidate, y)
    #         if is_cache[0]:
    #             continue
    #         if new_loss < loss:
    #             x_adv = x_adv_candidate.clone()
    #             loss = new_loss
    #         pbar.set_description(
    #             f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
    #         if loss == 0:
    #             assert torch.max(torch.abs(x_adv - x)) <= self.attack_config["eps"] + 10 ** -4
    #             return x_adv
    #     return x

    def p_selection(self, step):
        step = int(step / self.attack_config["max_iter"] * 10000)
        if 10 < step <= 50:
            p = self.attack_config["p_init"] / 2
        elif 50 < step <= 200:
            p = self.attack_config["p_init"] / 4
        elif 200 < step <= 500:
            p = self.attack_config["p_init"] / 8
        elif 500 < step <= 1000:
            p = self.attack_config["p_init"] / 16
        elif 1000 < step <= 2000:
            p = self.attack_config["p_init"] / 32
        elif 2000 < step <= 4000:
            p = self.attack_config["p_init"] / 64
        elif 4000 < step <= 6000:
            p = self.attack_config["p_init"] / 128
        elif 6000 < step <= 8000:
            p = self.attack_config["p_init"] / 256
        elif 8000 < step <= 10000:
            p = self.attack_config["p_init"] / 512
        else:
            p = self.attack_config["p_init"]
        return p

    def margin_loss(self, x, y):
        logits, is_cache = self.model(x)
        probs = torch.softmax(logits, dim=1)
        top_2_probs, top_2_classes = torch.topk(probs, 2)
        if top_2_classes[:, 0] != y:
            return 0, is_cache
        else:
            return top_2_probs[:, 0] - top_2_probs[:, 1], is_cache

    def attack_untargeted(self, x, y):
        dim = torch.prod(torch.tensor(x.shape[1:]))

        # Initialize adversarial example
        pert = torch.tensor(np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
                                             size=[x.shape[0], x.shape[1], 1, x.shape[3]])).float().to(x.device)
        x_adv = torch.clamp(x + pert, 0, 1)
        loss, is_cache = self.margin_loss(x_adv, y)
        if self.attack_config["adaptive"]["bs_num_squares"]:
            ns = self.binary_search_num_squares(x, x_adv)
        else:
            ns = 1
        if self.attack_config["adaptive"]["bs_min_square_size"]:
            min_s = self.binary_search_min_square_size(x, x_adv, ns)
        else:
            min_s = 1

        pbar = tqdm(range(self.attack_config["max_iter"]))
        step_attempts = 0
        for t in pbar:
            # x_adv_candidate = x_adv.clone()
            s = int(min(max(torch.sqrt(self.p_selection(t) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
            if self.attack_config["adaptive"]["bs_min_square_size"]:
                s = max(s, min_s)
            x_adv_candidate = self.add_squares(x, x_adv, s, ns)

            step_attempts += 1
            new_loss, is_cache = self.margin_loss(x_adv_candidate, y)
            if is_cache[0] and step_attempts < self.attack_config["adaptive"]["max_step_attempts"]:
                pbar.set_description(
                    f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
                continue
            elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["max_step_attempts"]:
                self.end("Step movement failure.")
            step_attempts = 0
            if new_loss < loss:
                x_adv = x_adv_candidate.clone()
                loss = new_loss
            pbar.set_description(
                f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
            if loss == 0:
                assert torch.max(torch.abs(x_adv - x)) <= self.attack_config["eps"] + 10 ** -4
                return x_adv
        return x

    def add_squares(self, x, x_adv, s, num_squares):
        x_adv_candidate = x_adv.clone()
        for _ in range(num_squares):
            pert = x_adv_candidate - x

            center_h = torch.randint(0, x.shape[2] - s, size=(1,)).to(x.device)
            center_w = torch.randint(0, x.shape[3] - s, size=(1,)).to(x.device)
            x_window = x[:, :, center_h:center_h + s, center_w:center_w + s]
            x_adv_window = x_adv_candidate[:, :, center_h:center_h + s, center_w:center_w + s]

            while torch.sum(
                    torch.abs(
                        torch.clamp(
                            x_window + pert[:, :, center_h:center_h + s, center_w:center_w + s], 0, 1
                        ) -
                        x_adv_window)
                    < 10 ** -7) == x.shape[1] * s * s:
                pert[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(
                    np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
                                     size=[x.shape[1], 1, 1])).float().to(x.device)
            x_adv_candidate = torch.clamp(x + pert, 0, 1)
        return x_adv_candidate

    def binary_search_num_squares(self, x, x_adv):
        dim = torch.prod(torch.tensor(x.shape[1:]))
        lower = self.attack_config["adaptive"]["bs_num_squares_lower"]
        upper = self.attack_config["adaptive"]["bs_num_squares_upper"]
        ns = upper
        for _ in range(self.attack_config["adaptive"]["bs_num_squares_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_num_squares_sample_size"]):
                s = int(min(max(torch.sqrt(self.p_selection(0) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
                noisy_img = self.add_squares(x, x_adv, s, int(mid))
                probs, is_cache = self.model(noisy_img)
                if is_cache[0]:
                    cache_hits += 1
            if cache_hits / self.attack_config["adaptive"]["bs_num_squares_sample_size"] \
                    <= self.attack_config["adaptive"]["bs_num_squares_hit_rate"]:
                ns = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Num Squares : {ns:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_num_squares_sample_size']}")
        return int(ns)

    def binary_search_min_square_size(self, x, x_adv, num_squares):
        lower = self.attack_config["adaptive"]["bs_min_square_size_lower"]
        upper = self.attack_config["adaptive"]["bs_min_square_size_upper"]
        min_ss = upper
        for _ in range(self.attack_config["adaptive"]["bs_min_square_size_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_min_square_size_sample_size"]):
                noisy_img = self.add_squares(x, x_adv, int(mid), num_squares)
                probs, is_cache = self.model(noisy_img)
                if is_cache[0]:
                    cache_hits += 1
            if cache_hits / self.attack_config["adaptive"]["bs_min_square_size_sample_size"] \
                    <= self.attack_config["adaptive"]["bs_min_square_size_hit_rate"]:
                min_ss = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Min Square Size : {min_ss:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_min_square_size_sample_size']}")
        return int(min_ss)
