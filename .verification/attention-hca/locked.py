# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run local device diagnostics under the installed Goal Plus lock protocol."""

import os
import subprocess
import sys
import uuid

from goal_plus.infrastructure.state.resource_locks import verifier_resource_lock


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        raise SystemExit("usage: <goal-plus-python> -I locked.py -- command [args...]")
    physical_device = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not physical_device.isdigit():
        raise SystemExit("local device work requires exactly one assigned physical device in visibility")
    request_id = f"attention-local-{uuid.uuid4().hex}"
    with verifier_resource_lock(f"decode-attention-npu:{physical_device}", request_id=request_id):
        print(f"[LOCK] acquired request_id={request_id}", flush=True)
        completed = subprocess.run(sys.argv[2:], env=os.environ.copy(), check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
