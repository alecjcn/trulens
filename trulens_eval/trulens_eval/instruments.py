"""
# App Instrumentation

## Designs and Choices

### App Data

We collect app components and parameters by walking over its structure and
producing a json reprensentation with everything we deem relevant to track. The
function `util.py:jsonify` is the root of this process.

#### class/system specific

##### pydantic (langchain)

Classes inheriting `pydantic.BaseModel` come with serialization to/from json in
the form of `BaseModel.dict` and `BaseModel.parse`. We do not use the
serialization to json part of this capability as a lot of langchain components
are tripped to fail it with a "will not serialize" message. However, we use make
use of pydantic `fields` to enumerate components of an object ourselves saving
us from having to filter out irrelevant internals.

We make use of pydantic's deserialization, however, even for our own internal
structures (see `schema.py`).

##### dataclasses (no present users)

The built-in dataclasses package has similar functionality to pydantic but we
presently do not handle it as we have no use cases.

##### dataclasses_json (llama_index)

Work in progress.

##### generic python (portions of llama_index and all else)

#### TruLens-specific Data

In addition to collecting app parameters, we also collect:

- (subset of components) App class information:

    - This allows us to deserialize some objects. Pydantic models can be
      deserialized once we know their class and fields, for example.
    - This information is also used to determine component types without having
      to deserialize them first. 
    - See `schema.py:Class` for details.

### Functions/Methods

Methods and functions are instrumented by overwriting choice attributes in
various classes. 

#### class/system specific

##### pydantic (langchain)

Most if not all langchain components use pydantic which imposes some
restrictions but also provides some utilities. Classes inheriting
`pydantic.BaseModel` do not allow defining new attributes but existing
attributes including those provided by pydantic itself can be overwritten (like
dict, for example). Presently, we override methods with instrumented versions.

#### Alternatives

- `intercepts` package (see https://github.com/dlshriver/intercepts)

    Low level instrumentation of functions but is architecture and platform
    dependent with no darwin nor arm64 support as of June 07, 2023.

- `sys.setprofile` (see
  https://docs.python.org/3/library/sys.html#sys.setprofile)

    Might incur much overhead and all calls and other event types get
    intercepted and result in a callback.

- langchain/llama_index callbacks. Each of these packages come with some
  callback system that lets one get various intermediate app results. The
  drawbacks is the need to handle different callback systems for each system and
  potentially missing information not exposed by them.

### Calls

The instrumented versions of functions/methods record the inputs/outputs and
some additional data (see `schema.py:RecordAppCall`). As more than one
instrumented call may take place as part of a app invokation, they are collected
and returned together in the `calls` field of `schema.py:Record`.

Calls can be connected to the components containing the called method via the
`path` field of `schema.py:RecordAppCallMethod`. This class also holds
information about the instrumented method.

#### Call Data (Arguments/Returns)

The arguments to a call and its return are converted to json using the same
tools as App Data (see above).

#### Tricky

- The same method call with the same `path` may be recorded multiple times in a
  `Record` if the method makes use of multiple of its versions in the class
  hierarchy (i.e. an extended class calls its parents for part of its task). In
  these circumstances, the `method` field of `RecordAppCallMethod` will
  distinguish the different versions of the method.

- Thread-safety -- it is tricky to use global data to keep track of instrumented
  method calls in presence of multiple threads. For this reason we do not use
  global data and instead hide instrumenting data in the call stack frames of
  the instrumentation methods. See `util.py:get_first_local_in_call_stack.py`.

#### Threads

Threads do not inherit call stacks from their creator. This is a problem due to
our reliance on info stored on the stack. Therefore we have a limitation:

- **Limitation**: Threads need to be started using the utility class TP in order
  for instrumented methods called in a thread to be tracked. As we rely on call
  stack for call instrumentation we need to preserve the stack before a thread
  start which python does not do.  See `util.py:TP._thread_starter`.

#### Async

Similar to threads, code run as part of a `asyncio.Task` does not inherit the
stack of the creator. Our current solution instruments `asyncio.new_event_loop`
to make sure all tasks that get created in `async` track the stack of their
creator. This is done in `utils/python.py:_new_event_loop` . The function
`stack_with_tasks` is then used to integrate this information with the normal
caller stack when needed. This may cause incompatibility issues when other tools
use their own event loops or interfere with this instrumentation in other ways.
Note that some async functions that seem to not involve `Task` do use tasks,
such as `gather`.

- **Limitation**: `async.Tasks` must be created via our `task_factory` as per
  `utils/python.py:task_factory_with_stack`. This includes tasks created by
  function such as `gather`. This limitation is not expected to be a problem
  given our instrumentation except if other tools are used that modify `async`
  in some ways.

#### Limitations

- Threading and async limitations. See **Threads** and **Async** .

- If the same wrapped sub-app is called multiple times within a single call to
  the root app, the record of this execution will not be exact with regards to
  the path to the call information. All call paths will address the last subapp
  (by order in which it is instrumented). For example, in a sequential app
  containing two of the same app, call records will be addressed to the second
  of the (same) apps and contain a list describing calls of both the first and
  second.

- Some apps cannot be serialized/jsonized. Sequential app is an example. This is
  a limitation of langchain itself.

- Instrumentation relies on CPython specifics, making heavy use of the `inspect`
  module which is not expected to work with other Python implementations.

#### Alternatives

- langchain/llama_index callbacks. These provide information about component
  invocations but the drawbacks are need to cover disparate callback systems and
  possibly missing information not covered.

## To Decide / To discuss

### Mirroring wrapped app behaviour and disabling instrumentation

Should our wrappers behave like the wrapped apps? Current design is like this:

```python
chain = ... # some langchain chain

tru = Tru() truchain = tru.Chain(chain, ...)

plain_result = chain(...) # will not be recorded

plain_result = truchain(...) # will be recorded

plain_result, record = truchain.call_with_record(...) # will be recorded, and
you get the record too
```

The problem with the above is that "call_" part of "call_with_record" is
langchain specific and implicitly so is __call__ whose behaviour we are
replicating in TruChain. Other wrapped apps may not implement their core
functionality in "_call" or "__call__".

Alternative #1:

```python

plain_result = chain(...) # will not be recorded

truchain = tru.Chain(chain, ...)

with truchain.record() as recorder:
    plain_result = chain(...) # will be recorded

records = recorder.records # can get records

truchain(...) # NOT SUPPORTED, use chain instead
```

Here we have the benefit of not having a special method for each app type like
`call_with_record`. We instead use a context to indicate that we want to collect
records and retrieve them afterwards.

### Calls: Implementation Details

Our tracking of calls uses instrumentated versions of methods to manage the
recording of inputs/outputs. The instrumented methods must distinguish
themselves from invocations of apps that are being tracked from those not being
tracked, and of those that are tracked, where in the call stack a instrumented
method invocation is. To achieve this, we rely on inspecting the python call
stack for specific frames:

- Root frame -- A tracked invocation of an app starts with one of the
  main/"root" methods such as call or query. These are the bottom of the
  relevant stack we use to manage the tracking of subsequent calls. Further
  calls to instrumented methods check for the root method in the call stack and
  retrieve the collection where data is to be recorded.
  
- Prior frame -- Each instrumented call also searches for the topmost
  instrumented call (except itself) in the stack to check its immediate caller
  (by immediate we mean only among instrumented methods) which forms the basis
  of the stack information recorded alongside the inputs/outputs.

#### Drawbacks

- Python call stacks are implementation dependent and we do not expect to operate
  on anything other than CPython.

- Python creates a fresh empty stack for each thread. Because of this, we need
  special handling of each thread created to make sure it keeps a hold of the
  stack prior to thread creation. Right now we do this in our threading utility
  class TP but a more complete solution may be the instrumentation of
  threading.Thread class.

- We require a root method to be placed on the stack to indicate the start of
  tracking. We therefore cannot implement something like a context-manager-based
  setup of the tracking system as suggested in the "To discuss" above.

  TODO: ROOTLESS

#### Alternatives

- `contextvars` -- langchain uses these to manage contexts such as those used for
  instrumenting/tracking LLM usage. These can be used to manage call stack
  information like we do. The drawback is that these are not threadsafe or at
  least need instrumenting thread creation. We have to do a similar thing by
  requiring threads created by our utility package which does stack management
  instead of contextvar management.

"""

from datetime import datetime
import inspect
from inspect import BoundArguments
from inspect import Signature
from inspect import signature
import logging
import os
from pprint import PrettyPrinter
import threading as th
import traceback
from typing import (
    Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple
)

from pydantic import BaseModel

from trulens_eval.feedback import Feedback
from trulens_eval.schema import Cost
from trulens_eval.schema import Perf
from trulens_eval.schema import Query
from trulens_eval.schema import RecordAppCall
from trulens_eval.schema import RecordAppCallMethod
from trulens_eval.util import _safe_getattr
from trulens_eval.util import dict_merge_with
from trulens_eval.util import get_all_local_in_call_stack
from trulens_eval.util import get_first_local_in_call_stack
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.util import Method

logger = logging.getLogger(__name__)
pp = PrettyPrinter()


class WithInstrumentCallbacks:
    """
    Callbacks invoked by Instrument during instrumentation or when instrumented
    methods are called. Needs to be mixed into App.
    """

    # Called during instrumentation.
    def _on_method_instrumented(
        self, obj: object, func: Callable, path: JSONPath
    ):
        """
        Called by instrumentation system for every function requested to be
        instrumented. Given are the object of the class in which `func` belongs
        (i.e. the "self" for that function), the `func` itsels, and the `path`
        of the owner object in the app hierarchy.
        """

        raise NotImplementedError

    # Called during invocation.
    def _get_method_path(self, obj: object, func: Callable) -> JSONPath:
        """
        Get the path of the instrumented function `func`, a member of the class
        of `obj` relative to this app.
        """

        raise NotImplementedError

    # WithInstrumentCallbacks requirement
    def _get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, JSONPath]]:
        """
        Get the methods (rather the inner functions) matching the given `func`
        and the path of each.
        """

        raise NotImplementedError

    # Called during invocation.
    def _on_new_record(self, func):
        """
        Called by instrumented methods in cases where they cannot find a record
        call list in the stack. If we are inside a context manager, return a new
        call list.
        """
        # TODO: ROOTLESS

        raise NotImplementedError

    # Called during invocation.
    def _on_add_record(
        self, record: Sequence[RecordAppCall], func: Callable, sig: Signature,
        bindings: BoundArguments, start_time, end_time, ret: Any, error: Any,
        cost: Cost
    ):
        """
        Called by instrumented methods if they use _new_record to construct a
        record call list. 
        """
        # TODO: ROOTLESS

        raise NotImplementedError


class Instrument(object):
    # TODO: might have to be made serializable soon.

    # Attribute name to be used to flag instrumented objects/methods/others.
    INSTRUMENT = "__tru_instrumented"

    # TODO: ROOTLESS
    # APPS = "__tru_apps"

    class Default:
        # Default instrumentation configuration. Additional components are
        # included in subclasses of `Instrument`.

        # Modules (by full name prefix) to instrument.
        MODULES = {"trulens_eval."}

        # Classes to instrument.
        CLASSES = set([Feedback])

        # Methods to instrument. Methods matching name have to pass the filter
        # to be instrumented. TODO: redesign this to be a dict with classes
        # leading to method names instead.
        METHODS = {"__call__": lambda o: isinstance(o, Feedback)}

    def to_instrument_object(self, obj: object) -> bool:
        """
        Determine whether the given object should be instrumented.
        """

        # NOTE: some classes do not support issubclass but do support
        # isinstance. It is thus preferable to do isinstance checks when we can
        # avoid issublcass checks.
        return any(isinstance(obj, cls) for cls in self.include_classes)

    def to_instrument_class(self, cls: type) -> bool:  # class
        """
        Determine whether the given class should be instrumented.
        """

        return any(issubclass(cls, parent) for parent in self.include_classes)

    def to_instrument_module(self, module_name: str) -> bool:
        """
        Determine whether a module with the given (full) name should be
        instrumented.
        """

        return any(
            module_name.startswith(mod2) for mod2 in self.include_modules
        )

    def __init__(
        self,
        root_methods: Optional[Set[Callable]] = None,
        include_modules: Iterable[str] = [],
        include_classes: Iterable[type] = [],
        include_methods: Dict[str, Callable] = {},
        callbacks: WithInstrumentCallbacks = None
    ):
        self.root_methods = root_methods or set([])

        self.include_modules = Instrument.Default.MODULES.union(
            set(include_modules)
        )

        self.include_classes = Instrument.Default.CLASSES.union(
            set(include_classes)
        )

        self.include_methods = dict_merge_with(
            dict1=Instrument.Default.METHODS,
            dict2=include_methods,
            merge=lambda f1, f2: lambda o: f1(o) or f2(o)
        )

        self.callbacks = callbacks

    def tracked_method_wrapper(
        self, query: Query, func: Callable, method_name: str, cls: type,
        obj: object
    ):
        """
        Instrument a method to capture its inputs/outputs/errors.
        """

        assert self.root_methods is not None, "Cannot instrument method without `root_methods`."

        assert not hasattr(
            func, "__func__"
        ), "Function expected but method received."

        if hasattr(func, Instrument.INSTRUMENT):
            logger.debug(f"\t\t\t{query}: {func} is already instrumented")

            # Notify the app instrumenting this method where it is located. Note
            # we store the method being instrumented in the attribute
            # Instrument.INSTRUMENT of the wrapped variant.
            original_func = getattr(func, Instrument.INSTRUMENT)
            self.callbacks._on_method_instrumented(
                obj, original_func, path=query
            )

            return func

            # TODO: How to consistently address calls to chains that appear more
            # than once in the wrapped chain or are called more than once.

        else:
            # Notify the app instrumenting this method where it is located:

            self.callbacks._on_method_instrumented(obj, func, path=query)

        logger.debug(f"\t\t\t{query}: instrumenting {method_name}={func}")

        sig = signature(func)

        async def awrapper(*args, **kwargs):
            # TODO: figure out how to have less repetition between the async and
            # sync versions of this method.

            logger.debug(f"{query}: calling async instrumented method {func}")

            # If not within a root method, call the wrapped function without
            # any recording.

            def find_root_methods(f):
                return id(f) in set(
                    [id(rm.__code__) for rm in self.root_methods]
                )  # or id(f) == id(wrapper.__code__) # TODO ROOTLESS

            # Look up whether the root instrumented method was called earlier in
            # the stack and "record_and_app" variable was defined there. Will
            # use that for recording the wrapped call.
            records_and_apps = list(
                get_all_local_in_call_stack(
                    key="record_and_app", func=find_root_methods, offset=1
                )
            )
            """
            # TODO: ROOTLESS

            is_root_call = False

            if len(records_and_apps) == 0:
                # If this is the first instrumented method in the stack, check
                # that the app wants it recorded.
                records_and_apps = self.on_new_record(func)

                # If so, indicate that this is a root instrumented call.
                is_root_call = True
            """

            if records_and_apps is None or len(records_and_apps) == 0:
                # Otherwise return result without instrumentation.

                logger.debug(f"{query}: no record found, not recording.")

                return await func(*args, **kwargs)

            # Otherwise keep track of inputs and outputs (or exception).

            error = None
            rets = None

            def find_instrumented(f):
                return id(f) in [id(awrapper.__code__)]

            # If a wrapped method was called in this call stack, get the prior
            # calls from this variable. Otherwise create a new chain stack. As
            # another wrinke, the addresses of methods in the stack may vary
            # from app to app that are watching this method. Hence we index the
            # stacks by id of the call record list which is unique to each app.
            pstacks = get_first_local_in_call_stack(
                key="stacks", func=find_instrumented, offset=1
            )
            # Note: Empty dict is false-ish.
            if pstacks is None:
                pstacks = dict()

            # My own stacks to be looked up by further subcalls by the logic
            # right above. We make a copy here since we need subcalls to access
            # it but we don't want them to modify it.
            stacks = dict()

            start_time = None
            end_time = None

            bindings = None

            # Prepare stacks with call information of this wrapped method so
            # subsequent (inner) calls will see it. For every root_method in the
            # call stack, we make a call record to add to the existing list
            # found in the stack. Path stored in `query` of this method may
            # differ between apps that use it so we have to create a seperate
            # frame identifier for each, and therefore the stack. We also need
            # to use a different stack for the same reason. We index the stack
            # in `stacks` via id of the (unique) list `record`.

            for record, app in records_and_apps:
                # Get record and app that has instrumented this method.

                rid = id(record)

                # The path to this method according to the app.
                path = app._get_method_path(
                    args[0], func
                )  # hopefully args[0] is self

                if path is None:
                    logger.warning(
                        f"App of type {type(app)} no longer knows about Object 0x{id(args[0]):x} method {func}."
                    )
                    continue

                if rid not in pstacks:
                    # If we are the first instrumented method in the chain
                    # stack, make a new stack tuple for subsequent deeper calls
                    # (if any) to look up.
                    stack = ()
                else:
                    stack = pstacks[rid]

                frame_ident = RecordAppCallMethod(
                    path=path, method=Method.of_method(func, obj=obj, cls=cls)
                )

                stack = stack + (frame_ident,)

                stacks[rid] = stack  # for deeper calls to get

                # Now we will call the wrapped method. We only do so once.

                # Start of run once condition.
                if start_time is None:
                    start_time = datetime.now()

                    try:
                        # Using sig bind here so we can produce a list of key-value
                        # pairs even if positional arguments were provided.
                        bindings: BoundArguments = sig.bind(*args, **kwargs)
                        """
                        # TODO: ROOTLESS
                        # If this is a root call (first instrumented method), also track
                        # costs:
                        cost: Cost = None
                        if is_root_call:
                            rets, cost = Endpoint.track_all_costs_tally(
                                lambda: func(*bindings.args, **bindings.kwargs)
                            )
                        else:
                        """

                        rets = await func(*bindings.args, **bindings.kwargs)

                        end_time = datetime.now()

                    except BaseException as e:
                        end_time = datetime.now()
                        error = e
                        error_str = str(e)

                        logger.error(
                            f"Error calling wrapped function {func.__name__}."
                        )
                        logger.error(traceback.format_exc())

                    # Done running the wrapped function. Lets collect the results.
                    # Create common information across all records.

                    # Don't include self in the recorded arguments.
                    nonself = {
                        k: jsonify(v) for k, v in (
                            bindings.arguments.items(
                            ) if bindings is not None else {}
                        ) if k != "self"
                    }

                    row_args = dict(
                        args=nonself,
                        perf=Perf(start_time=start_time, end_time=end_time),
                        pid=os.getpid(),
                        tid=th.get_native_id(),
                        rets=rets,
                        error=error_str if error is not None else None
                    )

                # End of run once condition.

                # Note that only the stack differs between each of the records in this loop.
                row_args['stack'] = stack
                row = RecordAppCall(**row_args)

                record.append(row)
                """
                # TODO: ROOTLESS
                if is_root_call:
                    # If this is a root call, notify app to add the completed record
                    # into its containers:
                    self.on_add_record(record, func, sig, bindings, cost)
                """

            if error is not None:
                raise error

            return rets

        def wrapper(*args, **kwargs):
            # TODO: figure out how to have less repetition between the async and
            # sync versions of this method.

            logger.debug(f"{query}: calling instrumented method {func}")

            # If not within a root method, call the wrapped function without
            # any recording.

            def find_root_methods(f):
                return id(f) in set(
                    [id(rm.__code__) for rm in self.root_methods]
                )  # or id(f) == id(wrapper.__code__) # TODO ROOTLESS

            # Look up whether the root instrumented method was called earlier in
            # the stack and "record_and_app" variable was defined there. Will
            # use that for recording the wrapped call.
            records_and_apps = list(
                get_all_local_in_call_stack(
                    key="record_and_app", func=find_root_methods, offset=1
                )
            )
            """
            # TODO: ROOTLESS

            is_root_call = False

            if len(records_and_apps) == 0:
                # If this is the first instrumented method in the stack, check
                # that the app wants it recorded.
                records_and_apps = self.on_new_record(func)

                # If so, indicate that this is a root instrumented call.
                is_root_call = True
            """

            if records_and_apps is None or len(records_and_apps) == 0:
                # Otherwise return result without instrumentation.

                logger.debug(f"{query}: no record found, not recording.")

                return func(*args, **kwargs)

            # Otherwise keep track of inputs and outputs (or exception).

            error = None
            rets = None

            def find_instrumented(f):
                return id(f) in [id(wrapper.__code__)]

            # If a wrapped method was called in this call stack, get the prior
            # calls from this variable. Otherwise create a new chain stack. As
            # another wrinke, the addresses of methods in the stack may vary
            # from app to app that are watching this method. Hence we index the
            # stacks by id of the call record list which is unique to each app.
            pstacks = get_first_local_in_call_stack(
                key="stacks", func=find_instrumented, offset=1
            )
            # Note: Empty dict is false-ish.
            if pstacks is None:
                pstacks = dict()

            # My own stacks to be looked up by further subcalls by the logic
            # right above. We make a copy here since we need subcalls to access
            # it but we don't want them to modify it.
            stacks = dict()

            start_time = None
            end_time = None

            bindings = None

            # Prepare stacks with call information of this wrapped method so
            # subsequent (inner) calls will see it. For every root_method in the
            # call stack, we make a call record to add to the existing list
            # found in the stack. Path stored in `query` of this method may
            # differ between apps that use it so we have to create a seperate
            # frame identifier for each, and therefore the stack. We also need
            # to use a different stack for the same reason. We index the stack
            # in `stacks` via id of the (unique) list `record`.

            for record, app in records_and_apps:
                # Get record and app that has instrumented this method.

                rid = id(record)

                # The path to this method according to the app.
                path = app._get_method_path(
                    args[0], func
                )  # args[0] is owner of wrapped method, hopefully

                if path is None:
                    logger.warning(
                        f"App of type {type(app)} no longer knows about Object 0x{id(args[0]):x} method {func}."
                    )
                    continue

                if rid not in pstacks:
                    # If we are the first instrumented method in the chain
                    # stack, make a new stack tuple for subsequent deeper calls
                    # (if any) to look up.
                    stack = ()
                else:
                    stack = pstacks[rid]

                frame_ident = RecordAppCallMethod(
                    path=path, method=Method.of_method(func, obj=obj, cls=cls)
                )

                stack = stack + (frame_ident,)

                stacks[rid] = stack  # for deeper calls to get

                # Now we will call the wrapped method. We only do so once.

                # Start of run once condition.
                if start_time is None:
                    start_time = datetime.now()

                    try:
                        # Using sig bind here so we can produce a list of key-value
                        # pairs even if positional arguments were provided.
                        bindings: BoundArguments = sig.bind(*args, **kwargs)
                        """
                        # TODO: ROOTLESS
                        # If this is a root call (first instrumented method), also track
                        # costs:
                        cost: Cost = None
                        if is_root_call:
                            rets, cost = Endpoint.track_all_costs_tally(
                                lambda: func(*bindings.args, **bindings.kwargs)
                            )
                        else:
                        """

                        rets = func(*bindings.args, **bindings.kwargs)

                        end_time = datetime.now()

                    except BaseException as e:
                        end_time = datetime.now()
                        error = e
                        error_str = str(e)

                        logger.error(
                            f"Error calling wrapped function {func.__name__}."
                        )
                        logger.error(traceback.format_exc())

                    # Done running the wrapped function. Lets collect the results.
                    # Create common information across all records.

                    # Don't include self in the recorded arguments.
                    nonself = {
                        k: jsonify(v) for k, v in (
                            bindings.arguments.items(
                            ) if bindings is not None else {}
                        ) if k != "self"
                    }

                    row_args = dict(
                        args=nonself,
                        perf=Perf(start_time=start_time, end_time=end_time),
                        pid=os.getpid(),
                        tid=th.get_native_id(),
                        rets=rets,
                        error=error_str if error is not None else None
                    )

                # End of run once condition.

                # Note that only the stack differs between each of the records in this loop.
                row_args['stack'] = stack
                row = RecordAppCall(**row_args)

                record.append(row)
                """
                # TODO: ROOTLESS
                if is_root_call:
                    # If this is a root call, notify app to add the completed record
                    # into its containers:
                    self.on_add_record(record, func, sig, bindings, cost)
                """

            if error is not None:
                raise error

            return rets

        w = wrapper
        if inspect.iscoroutinefunction(func):
            w = awrapper

        # Indicate that the wrapper is an instrumented method so that we dont
        # further instrument it in another layer accidentally.
        setattr(w, Instrument.INSTRUMENT, func)

        w.__name__ = func.__name__

        # Add a list of apps that want to record calls to this method starting
        # with self.
        # setattr(w, Instrument.APPS, [self])
        # TODO: ROOTLESS

        # NOTE(piotrm): This is important; langchain checks signatures to adjust
        # behaviour and we need to match. Without this, wrapper signatures will
        # show up only as *args, **kwargs .
        w.__signature__ = inspect.signature(func)

        return w

    def instrument_method(self, method_name, obj, query):
        cls = type(obj)

        logger.debug(f"{query}: instrumenting {method_name} on obj {obj}")

        for base in list(cls.__mro__):
            logger.debug(f"\t{query}: instrumenting base {base.__name__}")

            for method_name in [method_name]:

                if hasattr(base, method_name):
                    original_fun = getattr(base, method_name)

                    logger.debug(
                        f"\t\t{query}: instrumenting {base.__name__}.{method_name}"
                    )
                    setattr(
                        base, method_name,
                        self.tracked_method_wrapper(
                            query=query,
                            func=original_fun,
                            method_name=method_name,
                            cls=base,
                            obj=obj
                        )
                    )

    def instrument_object(self, obj, query: Query, done: Set[int] = None):

        done = done or set([])

        cls = type(obj)

        logger.debug(
            f"{query}: instrumenting object at {id(obj):x} of class {cls.__name__} with mro:\n\t"
            + '\n\t'.join(map(str, cls.__mro__))
        )

        if id(obj) in done:
            logger.debug(f"\t{query}: already instrumented")
            return

        done.add(id(obj))

        # NOTE: We cannot instrument chain directly and have to instead
        # instrument its class. The pydantic BaseModel does not allow instance
        # attributes that are not fields:
        # https://github.com/pydantic/pydantic/blob/11079e7e9c458c610860a5776dc398a4764d538d/pydantic/main.py#LL370C13-L370C13
        # .

        for base in list(cls.__mro__):
            # Some top part of mro() may need instrumentation here if some
            # subchains call superchains, and we want to capture the
            # intermediate steps. On the other hand we don't want to instrument
            # the very base classes such as object:
            if not self.to_instrument_module(base.__module__):
                continue

            try:
                if not self.to_instrument_class(base):
                    continue
            except Exception:
                # subclass check may raise exception
                continue

            logger.debug(f"\t{query}: instrumenting base {base.__name__}")

            for method_name in self.include_methods:

                if hasattr(base, method_name):
                    check_class = self.include_methods[method_name]
                    if not check_class(obj):
                        continue
                    original_fun = getattr(base, method_name)

                    # Sometimes the base class may be in some module but when a
                    # method is looked up from it, it actually comes from some
                    # other, even baser class which might come from builtins
                    # which we want to skip instrumenting.
                    if hasattr(original_fun, "__self__"):
                        if not self.to_instrument_module(
                                original_fun.__self__.__class__.__module__):
                            continue
                    else:
                        # Determine module here somehow.
                        pass

                    logger.debug(f"\t\t{query}: instrumenting {method_name}")
                    setattr(
                        base, method_name,
                        self.tracked_method_wrapper(
                            query=query,
                            func=original_fun,
                            method_name=method_name,
                            cls=base,
                            obj=obj
                        )
                    )

        if isinstance(obj, BaseModel):

            for k in obj.__fields__:
                # NOTE(piotrm): may be better to use inspect.getmembers_static .
                v = getattr(obj, k)

                if isinstance(v, str):
                    pass

                elif self.to_instrument_module(type(v).__module__):
                    self.instrument_object(obj=v, query=query[k], done=done)

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if any(isinstance(sv, cls)
                               for cls in self.include_classes):
                            self.instrument_object(
                                obj=sv, query=query[k][i], done=done
                            )

                # TODO: check if we want to instrument anything in langchain not
                # accessible through __fields__ .

        elif self.to_instrument_object(obj):
            # If an object is not a recognized container type, we check that it
            # is meant to be instrumented and if so, we  walk over it manually.
            # NOTE: llama_index objects are using dataclasses_json but most do
            # not so this section applies.

            for k in dir(obj):
                if k.startswith("_") and k[1:] in dir(obj):
                    # Skip those starting with _ that also have non-_ versions.
                    continue

                sv = _safe_getattr(obj, k)

                if self.to_instrument_object(sv):
                    self.instrument_object(obj=sv, query=query[k], done=done)

        else:
            logger.debug(
                f"{query}: Do not know how to instrument object of type {cls}."
            )
