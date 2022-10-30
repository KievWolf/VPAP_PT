from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
import trimesh
import trimesh.transformations as tra
import surface_normal
import copy
import os
from easydict import EasyDict as edict
import yaml
from vpap_data_reader import regularize_pc_point_count

from train import build_evaluator_ops, build_vae_ops

def choose_vpap_pt_better_than_threshold(eulers, translations, probs, threshold=0.7):
    """
      Обираємо сети мереж, які мають бали, вищі за вхідний поріг.
    """
    print('choose_better_than_threshold threshold=', threshold)
    return np.asarray(probs >= threshold, dtype=np.float32)

def choose_vpap_pt_better_than_threshold_in_sequence(eulers, translations, probs, threshold=0.7):
    """
      Обираємо сети мереж з максимальним балом у послідовності уточнення сету.
    """
    output = np.zeros(probs.shape, dtype=np.float32)
    max_index = np.argmax(probs, 0)
    max_value = np.max(probs, 0)
    # print(max_value)
    # print(max_index)
    for i in range(probs.shape[1]):
        if max_value[i] > threshold:
            output[max_index[i]][i] = 1.
    return output


def joint_config(vae_folder, evaluator_folder='', dataset_root_folder='', eval_split=''):
    """
     Об’єднуємо конфігурації vae і evaluator і переконуємося, що вони є
         сумісні між собою.    """

    print('--> завантаження {}'.format(os.path.join(vae_folder, 'args.yaml')))
    vae_cfg = vars(yaml.load(open(os.path.join(vae_folder, 'args.yaml'))))
    print('--> завантаження {}'.format(os.path.join(evaluator_folder, 'args.yaml')))
    evaluator_cfg = vars(yaml.load(open(os.path.join(evaluator_folder, 'args.yaml'))))

    args = copy.deepcopy(vae_cfg)
    for k in evaluator_cfg:
        if k in args:
            if k in set(['train_evaluator', 'ngpus', 'logdir']):
                continue
            if args[k] != evaluator_cfg[k]:
                print('conflicting key {}:'.format(k))
                print('  in vae, it is {}'.format(args[k]))
                print('  in evaluator, it is {}'.format(evaluator_cfg[k]))
        else:
            args[k] = evaluator_cfg[k]

    # додавання параметрів, які використовуються для висновків, а не для тестування.

    args['ngpus'] = 1
    args['gpu'] = 1
    args['vae_checkpoint_folder'] = os.path.join(vae_folder, 'tf_output')
    args['evaluator_checkpoint_folder'] = os.path.join(evaluator_folder, 'tf_output')
    args['sample_based_improvement'] = 1
    args['vpap_selection_mode'] = 'better_than_threshold_in_sequence'
    args['num_samples'] = 200
    args['eval_split'] = eval_split
    args['num_refine_steps'] = 20
    args['use_geometry_sampling'] = 0
    args['gan'] = 0
    if 'gan' in vae_cfg:
        args['gan'] = vae_cfg['gan']

    if dataset_root_folder != '':
        args['dataset_root_folder'] = dataset_root_folder

    return edict(args)


class VpapEstimator:
    """
      Включає код, який використовується для виконання висновку.
    """

    def __init__(self, cfg, scope=''):
        self._cfg = cfg
        self._scope = scope
        self._sample_based_improvements = self._cfg.sample_based_improvement
        self._num_samples = self._cfg.num_samples
        self._choose_fns = {
            'all': None,
            'better_than_threshold': choose_vpap_pt_better_than_threshold,
            'better_than_threshold_in_sequence': choose_vpap_pt_better_than_threshold_in_sequence,
        }

        self._data_dict = {
            'vae_pc': tf.placeholder(tf.float32, [self._num_samples, self._cfg.npoints, 3], name='sample_pc_input'),
            'evaluator_pc': tf.placeholder(tf.float32, [self._num_samples, self._cfg.npoints, 3],
                                           name='evaluator_pc_input'),
            'vae_pred/samples': tf.placeholder(tf.float32, [self._num_samples, self._cfg.latent_size],
                                               name='sample_latent_input'),
            'evaluator_vpap_eulers': [
                tf.placeholder(tf.float32, [self._num_samples], name='sample_ratation_{}'.format(i)) for i in
                range(3)],
            'evaluator_vpap_translations': tf.placeholder(tf.float32, [self._num_samples, 3],
                                                           name='sample_translation'),
        }

    def load_weights(
            self,
            sess,
            vae_checkpoint_folder=None,
            evaluator_checkpoint_folder=None,
            vae_scope='vae',
            evaluator_scope='evaluator',
    ):
        vae_checkpoint_folder = self._cfg.vae_checkpoint_folder if vae_checkpoint_folder is None else vae_checkpoint_folder
        evaluator_checkpoint_folder = self._cfg.evaluator_checkpoint_folder if evaluator_checkpoint_folder is None else evaluator_checkpoint_folder
        evaluator_vars = []
        vae_vars = []
        for v in tf.global_variables():
            if v.name.startswith(evaluator_scope):
                evaluator_vars.append(v)
            elif v.name.startswith(vae_scope):
                vae_vars.append(v)
            else:
                print(v.name)
        print('vae_vars = {} evaluator vars = {}'.format(len(vae_vars), len(evaluator_vars)))

        if vae_checkpoint_folder is not None:
            vae_saver = tf.train.Saver(var_list=vae_vars)
            print('latest vae checkpoint', tf.train.latest_checkpoint(vae_checkpoint_folder))
            vae_saver.restore(sess, tf.train.latest_checkpoint(vae_checkpoint_folder))

        evaluator_loader = tf.train.Saver(var_list=evaluator_vars)
        print('папка контрольної точки', evaluator_checkpoint_folder)
        print('остання контрольна точка оцінки', tf.train.latest_checkpoint(evaluator_checkpoint_folder))
        evaluator_loader.restore(sess, tf.train.latest_checkpoint(evaluator_checkpoint_folder))

    def create_network(self):
        self._cfg.is_training = 0
        self._cfg.train_evaluator = 1
        build_evaluator_ops(self._data_dict, self._cfg)
        self._cfg.train_evaluator = 0
        build_vae_ops(self._data_dict, self._cfg)

        print(self._data_dict.keys())

    def predict_and_refine_vpap_pt(
            self,
            sess,
            pc,
            num_refine_steps,
            latents,
            threshold,
            choose_fn,
            metadata,
            vpap_pt_rt=None,
    ):

        if vpap_pt_rt is None:
            batch_size = latents.shape[0]
        else:
            batch_size = vpap_pt_rt.shape[0]

        broadcast_pc = np.tile(np.expand_dims(pc, 0), [batch_size, 1, 1])

        if vpap_pt_rt is not None:
            vpap_eulers = np.asarray([tra.euler_from_matrix(g) for g in vpap_pt_rt], dtype=np.float32)
            vpap_translations = np.asarray([g[:3, 3] for g in vpap_pt_rt], dtype=np.float32)
        elif self._cfg.use_geometry_sampling:
            vpap_pt = surface_normal.propose_vpap_pt(pc, 0.01, latents.shape[0])
            vpap_eulers = np.asarray([tra.euler_from_matrix(g) for g in vpap_pt])
            vpap_translations = np.squeeze(vpap_pt[:, :3, 3])

            vae_feed_dict = {}
            vae_feed_dict[self._data_dict['vae_pc']] = broadcast_pc
            vae_feed_dict[self._data_dict['vae_pred/samples']] = latents

            print('запущено ініціалізацію...')
            vae_time = time.time()
            vpap_qt = sess.run(tf_vpap_qt, feed_dict=vae_feed_dict)
            vae_time = time.time() - vae_time
            print('це займе {} секунд'.format(vae_time))

            conversion_time = time.time()
            vpap_eulers, vpap_translations = self._convert_qt_to_rt(vpap_qt)
            print('QT у RT займе {} секунд'.format(time.time() - conversion_time))

        improved_probs = []
        improved_ts = []
        improved_eulers = []

        improved_ts.append(np.asarray(vpap_translations))
        improved_eulers.append([vpap_eulers[:, 0], vpap_eulers[:, 1], vpap_eulers[:, 2]])

        for refine_iter in range(num_refine_steps + 1):
            eval_and_improve = refine_iter < num_refine_steps
            if eval_and_improve:
                print('Покращення #{}'.format(refine_iter))


            improve_fn = self._improve_vpap_pt_sampling_based if self._sample_based_improvements else self._improve_vpap_pt_gradient_based
            improved_results = improve_fn(
                sess,
                broadcast_pc,
                improved_eulers[-1],
                improved_ts[-1],
                eval_and_improve=eval_and_improve,
                metadata=metadata,
            )
            if eval_and_improve:
                new_success, new_t, new_euler = improved_results
                improved_ts.append(new_t)
                improved_eulers.append(new_euler)
                improved_probs.append(new_success)
            else:
                new_success = improved_results
                improved_probs.append(new_success)

        improved_probs = np.asarray(improved_probs)
        improved_ts = np.asarray(improved_ts)

        for i in range(len(improved_eulers)):
            improved_eulers[i] = np.concatenate([np.expand_dims(x, -1) for x in improved_eulers[i]], axis=-1)
            improved_eulers[i] = np.asarray(improved_eulers[i], dtype=np.float32)

        improved_eulers = np.asarray(improved_eulers)
        print(improved_probs.shape)
        improved_probs = np.squeeze(improved_probs, -1)

        print(improved_eulers.shape, improved_ts.shape, improved_probs.shape)
        if choose_fn is None:
            selection_mask = np.ones(improved_probs.shape, dtype=np.float32)
        else:
            print(threshold)
            selection_mask = choose_fn(improved_eulers, improved_ts, improved_probs, threshold=threshold)

        return improved_probs, improved_ts, improved_eulers, selection_mask

    def _improve_vpap_pt_gradient_based(self, sess, pc, euler_angles, translation, eval_and_improve, metadata):

        assert (isinstance(euler_angles, list))
        assert (len(euler_angles) == 3)

        fd = {}
        fd[self._data_dict['evaluator_pc']] = pc
        for i in range(3):
            fd[self._data_dict['evaluator_vpap_eulers'][i]] = euler_angles[i]

        fd[self._data_dict['evaluator_vpap_translations']] = translation

        sess_time = time.time()
        if eval_and_improve:
            current_success, delta = sess.run(
                [self._data_dict['evaluator_pred/success'], self._data_dict['evaluator_gradient']],
                feed_dict=fd
            )
        else:
            current_success = sess.run(self._data_dict['evaluator_pred/success'], feed_dict=fd)
            return current_success

        print(time.time() - sess_time)

        delta_t = delta[0]
        norm_t = np.maximum(np.sqrt(np.sum(delta_t * delta_t, -1)), 1e-6)
        alpha = np.minimum(0.01 / norm_t, 1.0)

        improved_angles = [euler_angles[i] + delta[i + 1] * alpha for i in range(3)]
        improved_translations = translation + np.expand_dims(alpha, -1) * delta_t

        return current_success, improved_translations, improved_angles

    def _improve_vpap_pt_sampling_based(
            self,
            sess,
            pc,
            euler_angles,
            translation,
            eval_and_improve,
            metadata
    ):
        assert (metadata is not None)
        assert ('type' in metadata)
        print(metadata['type'])
        ph_translation = self._data_dict['evaluator_vpap_translations']
        ph_rotations = self._data_dict['evaluator_vpap_eulers']

        fd = {}
        fd[self._data_dict['evaluator_pc']] = pc
        for i in range(3):
            fd[ph_rotations[i]] = euler_angles[i]

        fd[ph_translation] = translation
        tf_success = self._data_dict['evaluator_pred/success']

        if eval_and_improve:
            sampling_type = metadata['type']
            supported_samplings = ['metropolis']
            assert (sampling_type in supported_samplings)

            if sampling_type == 'metropolis':
                improve_time = time.time()
                delta_t = 2 * (np.random.rand(*translation.shape) - 0.5)
                delta_t *= 0.02
                delta_euler_angles = [(np.random.rand(*euler_angles[i].shape) - 0.5) * 2 * np.radians(0) for i in
                                      range(3)]

                perturbed_translation = translation + delta_t
                perturbed_euler_angles = [euler_angles[i] + delta_euler_angles[i] for i in range(3)]

                if 'last_success' not in metadata:
                    metadata['last_success'] = sess.run(tf_success, feed_dict=fd)

                fd[ph_translation] = perturbed_translation

                for i in range(3):
                    fd[ph_rotations[i]] = perturbed_euler_angles[i]

                sess_time = time.time()
                perturbed_success = sess.run(tf_success, feed_dict=fd)
                print(time.time() - sess_time)
                ratio = perturbed_success / np.maximum(metadata['last_success'], 0.0001)

                next_success = metadata['last_success'].copy()
                next_translation = translation.copy()
                next_euler_angles = copy.deepcopy(euler_angles)

                mask = np.random.rand(*ratio.shape) <= ratio
                assert (len(mask.shape) == 2 and mask.shape[1] == 1), mask.shape
                ind = np.where(mask)[0]

                for i in ind:
                    next_success[i, :] = perturbed_success[i, :]
                    next_translation[i, :] = perturbed_translation[i, :]
                    for j in range(3):
                        next_euler_angles[j][i] = perturbed_euler_angles[j][i]

                output = (copy.deepcopy(metadata['last_success']), next_translation, next_euler_angles)
                metadata['last_success'] = next_success
                print('час = ', improve_time - time.time())

                return output
            else:
                raise NotImplementedError('invalid sampling type {}'.format(sampling_type))


        else:
            eval_time = time.time()
            if 'last_success' in metadata:
                return metadata['last_success']

            current_success = sess.run(tf_success, feed_dict=fd)
            print('час виконання', time.time() - eval_time)
            return current_success

    def predict_vpap_pt(
            self,
            sess,
            input_pc,
            latents,
            num_refine_steps,
            return_vpap_indexes=False,
            base_to_camera_rt=None,
            vpap_pt_rt=None,
    ):
        normalize_pc_count = input_pc.shape[0] != self._cfg.npoints
        if normalize_pc_count:
            pc = regularize_pc_point_count(input_pc, self._cfg.npoints).copy()
        else:
            pc = input_pc.copy()

        pc_mean = np.mean(pc, 0)
        pc -= np.expand_dims(pc_mean, 0)

        batch_size = self._cfg.num_samples
        if vpap_pt_rt is not None:
            n = vpap_pt_rt.shape[0]
        else:
            n = latents.shape[0]

        output_vpap_pt = [ ]
        output_scores = [ ]
        output_latents = [ ]
        output_vpap_indexes = [ ]

        for i in range(0, n, self._cfg.num_samples):
            interval_index = range(i, min(i + batch_size, n))
            num_elems_added = 0
            if vpap_pt_rt is None:
                current_vpap_pt_rt = None
            else:
                current_vpap_pt_rt = vpap_pt_rt[interval_index]
                num_elems_added = batch_size - current_vpap_pt_rt.shape[0]
                extended_current_vpap_pt_rt = np.zeros((batch_size, 4, 4), dtype=np.float32)
                extended_current_vpap_pt_rt[:current_vpap_pt_rt.shape[0], :, :] = current_vpap_pt_rt
                current_vpap_pt_rt = extended_current_vpap_pt_rt
            if latents is None:
                current_latents = None
            else:
                current_latents = latents[interval_index]
                num_elems_added = batch_size - current_latents.shape[0]
                extended_current_latents = np.zeros((batch_size, self._cfg.latent_size))
                extended_current_latents[:current_latents.shape[0], :] = current_latents
                current_latents = extended_current_latents

            vpap_pt, scores, output_latents, vpap_indexes = self._predict_vpap_pt(
                sess,
                pc,
                current_latents,
                num_refine_steps,
                self._cfg.threshold,
                base_to_camera_rt=None,
                vpap_pt_rt=current_vpap_pt_rt,
                return_vpap_indexes=True,
            )
            for g, s, l, gi in zip(vpap_pt, scores, output_latents, vpap_indexes):
                if gi[1] < batch_size - num_elems_added:
                    unnormalized_g = g.copy()
                    unnormalized_g[:3, 3] += pc_mean
                    output_vpap_pt.append(unnormalized_g)
                    output_scores.append(s)
                    output_latents.append(l)
                    output_vpap_indexes.append(gi)

        if return_vpap_indexes:
            return output_vpap_pt, output_scores, output_latents, output_vpap_indexes

        return output_vpap_pt, output_scores, output_latents

    def _predict_vpap_pt(
            self,
            sess,
            pc,
            latents,
            num_refine_steps,
            threshold,
            return_vpap_indexes=False,
            base_to_camera_rt=None,
            vpap_pt_rt=None,
    ):

        metadata = {'type': 'metropolis'}
        vpap_pt = []
        scores = []
        output_latents = []

        improved_probs, improved_ts, improved_eulers, selection_mask = self._predict_and_refine_vpap_pt(
            sess,
            pc,
            num_refine_steps,
            latents,
            threshold=threshold,
            choose_fn=self._choose_fns[self._cfg.vpap_selection_mode],
            metadata=metadata,
            vpap_pt_rt=vpap_pt_rt,
        )

        refine_indexes, sample_indexes = np.where(selection_mask)

        if latents is None:
            output_latents = None

        vpap_indexes = []
        num_removed_vpap_pt = 0
        for refine_index, sample_index in zip(refine_indexes, sample_indexes):
            rt = tra.euler_matrix(*improved_eulers[refine_index, sample_index, :])
            rt[:3, 3] = improved_ts[refine_index, sample_index, :]
            if base_to_camera_rt is not None:
                dot = np.sum(rt[:3, 2] * -base_to_camera_rt[:3, 2])

                if dot < 0:
                    num_removed_vpap_pt += 1
                    continue

            scores.append(improved_probs[refine_index, sample_index])
            vpap_pt.append(rt.copy())
            if latents is not None:
                output_latents.append(latents[sample_index])
            vpap_indexes.append([int(refine_index), int(sample_index)])

        if num_removed_vpap_pt > 0:
            print('видалити {} vpap_pt у camera_to_base_rt'.format(num_removed_vpap_pt))

        if return_vpap_indexes:
            return vpap_pt, scores, output_latents, vpap_indexes

        return vpap_pt, scores, output_latents

    def _convert_qt_to_rt(self, vpap_pt):
        Rs = []
        Ts = []

        for g in vpap_pt:
            Rs.append(tra.euler_from_quaternion(g[:4]))
            Ts.append(g[4:])
        Rs = np.asarray(Rs)

        return Rs, Ts

    def sample_latents(self):
        if 'gan' in self._cfg and self._cfg['gan'] == 1:
            latents = np.random.rand(
                self._cfg.num_samples, self._cfg.latent_size)
        else:
            latents = np.random.rand(
                self._cfg.num_samples, self._cfg.latent_size) * 4 - 2
        return latents

