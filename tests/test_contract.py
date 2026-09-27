"""CPU regression checks for the repaired latent and random-state contracts."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the policy environment for tensor contract tests.")
class ContractTests(unittest.TestCase):
    def test_native_coordinate_conventions(self):
        from starVLA.model.modules.world_model.latent_contract import configure_coordinates
        sigma = torch.tensor([2.0, 3.0]).reshape(1, 2, 1, 1, 1)
        for native in ("divisor", "multiplier"):
            for mode in ("canonical", "legacy"):
                pipe = SimpleNamespace(
                    vae=SimpleNamespace(config=SimpleNamespace(
                        latents_mean=[0.0, 0.0], latents_std=[2.0, 3.0])),
                    latents_std=sigma if native == "divisor" else 1 / sigma,
                )
                _, divisor, detected = configure_coordinates(pipe, mode)
                expected = sigma if mode == "canonical" else 1 / sigma
                self.assertEqual(detected, native)
                self.assertTrue(torch.equal(divisor, expected))
                native_divisor = pipe.latents_std if native == "divisor" else 1 / pipe.latents_std
                self.assertTrue(torch.allclose(native_divisor, expected))

    def test_unknown_coordinates_are_rejected(self):
        from starVLA.model.modules.world_model.latent_contract import configure_coordinates
        pipe = SimpleNamespace(
            vae=SimpleNamespace(config=SimpleNamespace(latents_mean=[0.0], latents_std=[2.0])),
            latents_std=torch.tensor([7.0]).reshape(1, 1, 1, 1, 1),
        )
        with self.assertRaises(ValueError):
            configure_coordinates(pipe, "legacy")

    def test_checkpoint_metadata_cannot_be_overridden_silently(self):
        from starVLA.model.modules.world_model.latent_contract import prepare_checkpoint_config
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            Path(str(checkpoint) + ".metadata.json").write_text(
                json.dumps({"latent_normalization": "legacy"})
            )
            config = {"framework": {"name": "CosmoPredict25PerceiverVJEPA2AC",
                                    "future_image_generation": {"num_frames": 93}}}
            with self.assertRaises(ValueError):
                prepare_checkpoint_config(config, checkpoint, "canonical")
            with self.assertWarns(UserWarning):
                prepared = prepare_checkpoint_config(config, checkpoint, "legacy")
            self.assertEqual(prepared["framework"]["future_image_generation"]["num_frames"], 5)

    def test_temporal_extrapolation_requires_opt_in(self):
        from starVLA.model.modules.world_model.latent_contract import validate_generation_length
        self.assertEqual(validate_generation_length(5, 5), 5)
        with self.assertRaises(ValueError):
            validate_generation_length(93, 5)
        with self.assertRaises(ValueError):
            validate_generation_length(6, 5, True)
        self.assertEqual(validate_generation_length(93, 5, True), 93)

    def test_seeded_requests_do_not_change_the_outer_rng(self):
        from starVLA.model.modules.world_model.reproducibility import seeded_inference

        class Policy(torch.nn.Module):
            @seeded_inference
            def sample(self):
                return torch.randn(4)

        policy = Policy()
        state = torch.get_rng_state().clone()
        first = policy.sample(inference_seed=123)
        policy.sample(inference_seed=456)
        again = policy.sample(inference_seed=123)
        self.assertTrue(torch.equal(first, again))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for invalid in (True, -1, 2**63, 1.2):
            with self.assertRaises(ValueError):
                policy.sample(inference_seed=invalid)

    def test_evaluation_context_restores_mixed_modes_and_rng_on_error(self):
        from starVLA.model.modules.world_model.reproducibility import evaluation_context
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        model.train()
        model[0].eval()
        before_modes = [module.training for module in model.modules()]
        before_torch = torch.get_rng_state().clone()
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with evaluation_context(model):
                self.assertFalse(any(module.training for module in model.modules()))
                torch.rand(3)
                random.random()
                np.random.rand()
                raise RuntimeError("test exception")
        self.assertEqual(before_modes, [module.training for module in model.modules()])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        self.assertTrue(np.array_equal(before_numpy[1], np.random.get_state()[1]))
        self.assertEqual(before_numpy[2:], np.random.get_state()[2:])


if __name__ == "__main__":
    unittest.main()
