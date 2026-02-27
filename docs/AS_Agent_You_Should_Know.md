1. Core Philosophy

You must follow the principle of Occam’s Razor:

The simplest solution that satisfies the requirements is preferred.

Avoid unnecessary abstraction, over-engineering, excessive modularization, or speculative optimization.
Do not introduce complexity unless it is strictly required.

2. Code Simplicity Requirements

Prefer small, direct, readable functions

Avoid deeply nested logic

Avoid unnecessary class hierarchies

Avoid creating generic frameworks unless explicitly requested

Do not refactor working code into more complex patterns without justification

When implementing a function:

Solve only the stated problem

Do not add extra features

Do not anticipate future requirements unless explicitly instructed

3. File Creation Rules (Critical)

You must NOT create .md files unless explicitly requested.

Specifically:

Do not generate documentation files automatically

Do not create README files

Do not create design documents

Do not create architecture explanation files

Do not create summary or analysis markdown files

All explanations should be provided inline in chat unless file creation is explicitly requested.

Excessive file generation can destabilize the workspace and must be avoided.

4. Documentation and Comments (Required)

Every function and module must include clear comments.

Each function must include:

A brief description of purpose

Description of parameters

Description of return value

Any side effects (if applicable)

Example format:

def compute_loss(pred, target):
    """
    Compute mean squared error between prediction and target.

    Args:
        pred (Tensor): Model prediction tensor.
        target (Tensor): Ground truth tensor.

    Returns:
        Tensor: Scalar loss value.
    """

Each module should start with a short header explaining:

What the module does

What problem it solves

Any key assumptions

5. Complexity Control Rules

Avoid:

Overuse of decorators

Meta-programming

Dynamic code generation

Over-abstracted design patterns

Factory patterns unless necessary

Dependency injection frameworks unless requested

Prefer:

Explicit logic

Linear control flow

Readability over cleverness

Clarity over brevity

6. Modification Rules

When modifying existing code:

Preserve original structure unless change is necessary

Do not reorganize the entire file without request

Do not rename variables unless improving clarity

Do not split files unless explicitly instructed

7. Files/Code snippet Should Not Change
You should not change codes from /third_party, as they are officially correct codes, but you should understand deeply what they are doing.

8. Write Your Unit Test Codes:
When you help me coding, you should design and write your unit test codes into /test folder, everytime you debug or add new function or add new code snippet, you should run the unit tests.