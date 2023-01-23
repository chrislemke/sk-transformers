# Changelog

## [0.9.1](https://github.com/chrislemke/sk-transformers/compare/v0.9.0...v0.9.1) (2023-01-23)


### CI/CD

* add auto comment github action ([#58](https://github.com/chrislemke/sk-transformers/issues/58)) ([bed2782](https://github.com/chrislemke/sk-transformers/commit/bed2782594ac6881d4ac1f2e7643de4f293cf80b))
* improve github actions ([#64](https://github.com/chrislemke/sk-transformers/issues/64)) ([e1f07e1](https://github.com/chrislemke/sk-transformers/commit/e1f07e1b41e1ae81c72b7306c6323a54ff9d0319))

## [0.9.0](https://github.com/chrislemke/sk-transformers/compare/v0.8.0...v0.9.0) (2023-01-20)


### Features

* add ColumnEvalTransformer ([#52](https://github.com/chrislemke/sk-transformers/issues/52)) ([a03b079](https://github.com/chrislemke/sk-transformers/commit/a03b079d1818674c7115b4f3122656f0f1af1b1d))
* add string_splitter_transformer ([#53](https://github.com/chrislemke/sk-transformers/issues/53)) ([fdf89e1](https://github.com/chrislemke/sk-transformers/commit/fdf89e1dd9cb9de1348a9be11796a24023ec1817))


### Bug Fixes

* allow nans in query transformer ([b8bf874](https://github.com/chrislemke/sk-transformers/commit/b8bf8748124b12f634182af5660875f3b98e397c))


### Maintenance

* improve data check by only checking used columns ([#54](https://github.com/chrislemke/sk-transformers/issues/54)) ([ca450a4](https://github.com/chrislemke/sk-transformers/commit/ca450a4d69d0b9136996edf3dedb8a7e51b148d7))


### CI/CD

* add ability to run the Release action manually ([#56](https://github.com/chrislemke/sk-transformers/issues/56)) ([42f76a3](https://github.com/chrislemke/sk-transformers/commit/42f76a318e1202c6556b4b7f335cb855ec30368a))

## [0.8.0](https://github.com/chrislemke/sk-transformers/compare/v0.7.4...v0.8.0) (2023-01-18)


### Features

* add allowed_values_transformer ([#46](https://github.com/chrislemke/sk-transformers/issues/46)) ([2fe06f6](https://github.com/chrislemke/sk-transformers/commit/2fe06f6ebd688faa7bba7fb3a51b431fa4f83040))


### Documentation

* fix broken url ([a1279b4](https://github.com/chrislemke/sk-transformers/commit/a1279b4a6116f8580149070e6bde1231b0747971))


### Maintenance

* add init file for easier usage of transformers ([#45](https://github.com/chrislemke/sk-transformers/issues/45)) ([e1edb18](https://github.com/chrislemke/sk-transformers/commit/e1edb18c4a184e771de577eca6ab24c77fe38339))

## [0.7.4](https://github.com/chrislemke/sk-transformers/compare/v0.7.3...v0.7.4) (2023-01-16)


### Bug Fixes

* allow nan in some transformers ([#41](https://github.com/chrislemke/sk-transformers/issues/41)) ([d2a4fbf](https://github.com/chrislemke/sk-transformers/commit/d2a4fbff12bba82c0cc0077673f8ee5d3a6fcca9))


### Maintenance

* add more pre-commit hooks ([#40](https://github.com/chrislemke/sk-transformers/issues/40)) ([b716c44](https://github.com/chrislemke/sk-transformers/commit/b716c44693666fc64d30a1d15f861de6ab66d8d3))


### CI/CD

* add fast-forward merge action to pull requests ([#42](https://github.com/chrislemke/sk-transformers/issues/42)) ([41036b9](https://github.com/chrislemke/sk-transformers/commit/41036b95ec4f6af29844409b467deb17f597b92c))

## [0.7.3](https://github.com/chrislemke/sk-transformers/compare/v0.7.2...v0.7.3) (2023-01-12)


### Bug Fixes

* deep transformer check ready to transform fails ([91f9712](https://github.com/chrislemke/sk-transformers/commit/91f97120d04f724c0df9b6a9fb42b16d9bda5a28))
* issue with failing check_array if dataframe contains objects and nan ([b99a734](https://github.com/chrislemke/sk-transformers/commit/b99a7345d9fd2ac875e0576961cdb4f024b755b1))

## [0.7.2](https://github.com/chrislemke/sk-transformers/compare/v0.7.1...v0.7.2) (2023-01-10)


### Maintenance

* add training_objective argument to tovectransformer ([#36](https://github.com/chrislemke/sk-transformers/issues/36)) ([041f11f](https://github.com/chrislemke/sk-transformers/commit/041f11fd42437cae058c018b84b00455a705f175))

## [0.7.1](https://github.com/chrislemke/sk-transformers/compare/v0.7.0...v0.7.1) (2023-01-06)


### Documentation

* fix icon issue in readme.md ([2371493](https://github.com/chrislemke/sk-transformers/commit/237149335a1a7cc7453609ee4784a6cfbf606da9))

## [0.7.0](https://github.com/chrislemke/sk-transformers/compare/v0.6.3...v0.7.0) (2023-01-06)


### Features

* add left_join_transformer ([#29](https://github.com/chrislemke/sk-transformers/issues/29)) ([31fbde0](https://github.com/chrislemke/sk-transformers/commit/31fbde02aada7c81236d4775b9ccc7f29510ac2f))


### Bug Fixes

* add `numpy` as a possible prefix ([#31](https://github.com/chrislemke/sk-transformers/issues/31)) ([8fec7d3](https://github.com/chrislemke/sk-transformers/commit/8fec7d30b4ed8415b090182c7a27d09310f070a2))


### CI/CD

* fix code-cov action ([f2be192](https://github.com/chrislemke/sk-transformers/commit/f2be1920fb037a3b5e8e215347613035df4dd441))


### Maintenance

* general improvements ([85596c2](https://github.com/chrislemke/sk-transformers/commit/85596c2526accb6bda9e7e5efd959fbf8ea28588))
* use swifter to speed up pandas apply ([#28](https://github.com/chrislemke/sk-transformers/issues/28)) ([53684e9](https://github.com/chrislemke/sk-transformers/commit/53684e912fa752e0c2902e99b93fa45dacda2613))


### Documentation

* add new project logo ([#32](https://github.com/chrislemke/sk-transformers/issues/32)) ([93e277b](https://github.com/chrislemke/sk-transformers/commit/93e277b6b7c26e7fdb1919512bdc188b2d51254f))
* add playground notebook ([#30](https://github.com/chrislemke/sk-transformers/issues/30)) ([681eaf9](https://github.com/chrislemke/sk-transformers/commit/681eaf92e3cf41a9fb93b446b2b5c21877ddf5f1))

## [0.6.3](https://github.com/chrislemke/sk-transformers/compare/v0.6.2...v0.6.3) (2023-01-04)


### Features

* deep transformer module ([#24](https://github.com/chrislemke/sk-transformers/issues/24)) ([fd266cf](https://github.com/chrislemke/sk-transformers/commit/fd266cf10c629cc5c5d33528006480fd5094cc96))

## [0.6.2](https://github.com/chrislemke/sk-transformers/compare/v0.6.1...v0.6.2) (2022-12-22)


### Documentation

* improve contributing file ([3b5287b](https://github.com/chrislemke/sk-transformers/commit/3b5287b1cd326b65331b086b4d2c8275a3dd170a))


### CI/CD

* improve the release process ([096f3b2](https://github.com/chrislemke/sk-transformers/commit/096f3b2482688ab561bf3afa4b6d8b98e1736186))

## [0.6.1](https://github.com/chrislemke/sk-transformers/compare/v0.6.0...v0.6.1) (2022-12-22)


### Bug Fixes

* wrong formatting in readme.md ([425f4ed](https://github.com/chrislemke/sk-transformers/commit/425f4ed1cf173ffad7534dae035528bc2fa81072))


### Documentation

* improve the usability ([4190cd7](https://github.com/chrislemke/sk-transformers/commit/4190cd7c356b540e788dfe5930ab198a1c5a13fe))

## [0.6.0](https://github.com/chrislemke/sk-transformers/compare/v0.5.7...v0.6.0) (2022-12-22)


### Features

* add scripts to improve the release process 🚀 ([54a9dfe](https://github.com/chrislemke/sk-transformers/commit/54a9dfeda3c4448502206f5e3181f69da17df9a5))
