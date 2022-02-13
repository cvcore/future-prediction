from yacs.config import CfgNode
from futurepred.utils.logger import Logger
from pathlib import Path

logger = Logger.default_logger()

class MyCfgNode(CfgNode):

    def update_from_args(self, args):
        self.defrost()
        self.merge_from_file(args.config)
        self.EXPERIMENT_NAME = Path(args.config).stem
        if args.opts is not None and len(args.opts) != 0:
            assert len(args.opts) % 2 == 0, f"Number of key and value doesn't match for config!"
            logger.warning(f"The following options will be updated from CLI:")
            for i in range(0, len(args.opts), 2):
                logger.warning(f"{args.opts[i]} = {args.opts[i+1]}")
            self.merge_from_list(args.opts)
        self.freeze()
